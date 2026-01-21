"""
3D并行训练主脚本
DDP (数据并行) + Pipeline Parallelism (流水线并行) + Tensor Parallelism (张量并行)
"""
import os
import argparse
import json
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import deepspeed
from loguru import logger
from src.models.model import LLMModel
from src.dataset import SimpleTextDataset


def setup_distributed():
    """初始化分布式环境，确保MASTER_ADDR和MASTER_PORT环境变量被设置"""
    # 设置分布式训练的 master 地址和端口（如果未手动指定）
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    # 从环境变量获取rank和world_size
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def get_tp_group_and_rank(world_size: int, tp_size: int):
    """
    获取张量并行组和rank

    假设: world_size = 8, tp_size = 2, pp_size = 2
    则 dp_size = 8 / (2 * 2) = 2

    组织结构:
    DP0: [GPU0, GPU1] (TP组0, PP stage 0)
         [GPU2, GPU3] (TP组0, PP stage 1)
    DP1: [GPU4, GPU5] (TP组1, PP stage 0)
         [GPU6, GPU7] (TP组1, PP stage 1)
    """
    rank = dist.get_rank()

    # 简化处理: 假设只用TP，不用PP
    tp_rank = rank % tp_size
    tp_group_ranks = [i for i in range(rank - tp_rank, rank - tp_rank + tp_size)]

    # 创建TP进程组
    tp_group = dist.new_group(tp_group_ranks)

    return tp_group, tp_rank

def get_ds_config(args):
    """
    智能获取DeepSpeed配置：
    1. 优先读取命令行传入的 JSON 文件
    2. 如果没有文件，使用默认字典
    3. 强制使用命令行参数(如LR, batch_size)覆盖配置文件，避免不一致
    """
    ds_config = None
    
    # 1. 尝试读取配置文件
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        if int(os.environ.get('RANK', 0)) == 0:
            logger.info(f"Loaded DeepSpeed config from: {args.deepspeed_config}")
    
    # 2. 如果没有文件，构建默认配置
    if ds_config is None:
        if int(os.environ.get('RANK', 0)) == 0:
            logger.warning("No DeepSpeed config file provided, using internal default config.")
        ds_config = {
            "train_batch_size": args.batch_size * int(os.environ.get("WORLD_SIZE", 1)) * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": args.logging_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": args.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": args.warmup_steps,
                    "total_num_steps": args.max_steps
                }
            },
            "fp16": {"enabled": args.fp16},
            "zero_optimization": {
                "stage": args.zero_stage,
            }
        }

    # 3. 强制覆盖关键参数 (确保CLI参数生效)
    # 注意：这里会覆盖JSON里的设置，如果你希望JSON优先，请注释掉这些行
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    
    return ds_config


def create_model(args, tp_rank: int, tp_world_size: int):
    """创建模型"""
    model = LLMModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        dropout=args.dropout
    )

    return model


def train_with_deepspeed(args):
    """使用DeepSpeed进行3D并行训练"""
    print("=" * 50)
    print("Starting 3D Parallel Training with DeepSpeed")
    print("=" * 50)

    # 初始化DeepSpeed
    deepspeed.init_distributed()

    local_rank = int(args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size}, Local Rank: {local_rank}")

    # 创建模型
    model = LLMModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )

    if rank == 0:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # 创建数据集
    train_dataset = SimpleTextDataset(
        num_samples=args.num_samples,
        seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )

    # DeepSpeed配置
    ds_config = get_ds_config(args)

    # 初始化DeepSpeed引擎
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config
    )

    # 训练循环
    global_step = 0
    model_engine.train()

    if rank == 0:
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        print(f"Total samples: {len(train_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Total batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Total steps: {args.max_steps}")
        print("=" * 50 + "\n")

    epoch = 0
    while global_step < args.max_steps:
        epoch += 1

        if rank == 0:
            print(f"\nEpoch {epoch}")
            train_dataloader = tqdm(train_dataloader, desc=f"Training")

        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # 将数据移到GPU
            input_ids = batch['input_ids'].to(model_engine.device)
            attention_mask = batch['attention_mask'].to(model_engine.device)
            labels = batch['labels'].to(model_engine.device)
            
            if args.fp16:
                attention_mask = attention_mask.half()

            # 前向传播
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # 反向传播
            model_engine.backward(loss)

            # 更新参数
            model_engine.step()

            # 记录
            if loss is not None:
                epoch_loss += loss.item()

            global_step += 1

            # 日志
            if global_step % args.logging_steps == 0 and rank == 0:
                avg_loss = epoch_loss / args.logging_steps
                print(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                epoch_loss = 0.0

            # 保存检查点
            if global_step % args.save_steps == 0:
                dist.barrier()
                
                # DeepSpeed 内部会自动只让主进程写文件，其他进程配合传输数据
                model_engine.save_checkpoint(args.output_dir, tag=f"checkpoint-{global_step}")
                
                if rank == 0:
                    print(f"Checkpoint saved to {args.output_dir}/checkpoint-{global_step}")


            if global_step >= args.max_steps:
                break

    # 最终保存
    model_engine.save_checkpoint(args.output_dir, tag="final_model")

    if rank == 0:
        full_path = os.path.join(args.output_dir, "final_model")
        print(f"\nTraining completed! Final model saved to {full_path}")


def train_with_pytorch_ddp(args):
    """使用PyTorch DDP + 手动实现的TP进行训练"""
    print("=" * 50)
    print("Starting 3D Parallel Training with PyTorch DDP")
    print("=" * 50)

    # 初始化分布式
    local_rank, rank, world_size = setup_distributed()

    print(f"Rank {rank}/{world_size}, Local Rank: {local_rank}")

    # 计算TP配置
    tp_size = args.tp_size if args.tp_size > 0 else 1
    tp_rank = rank % tp_size

    # 创建模型
    model = create_model(args, tp_rank, tp_size)
    model = model.to(local_rank)

    if rank == 0:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
        print(f"TP size: {tp_size}, TP rank: {tp_rank}")

    # DDP包装
    model = DDP(model, device_ids=[local_rank])

    # 创建数据集和dataloader
    train_dataset = SimpleTextDataset(
        num_samples=args.num_samples,
        seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 训练循环
    global_step = 0
    model.train()

    if rank == 0:
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        print(f"Total samples: {len(train_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Total batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Total steps: {args.max_steps}")
        print("=" * 50 + "\n")

    epoch = 0
    optimizer.zero_grad()

    while global_step < args.max_steps:
        epoch += 1
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch}")

        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # 将数据移到GPU
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']

                # 梯度累积
                loss = loss / args.gradient_accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 累积梯度后更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += loss.item() * args.gradient_accumulation_steps

                # 日志
                if global_step % args.logging_steps == 0 and rank == 0:
                    avg_loss = epoch_loss / args.logging_steps
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                    epoch_loss = 0.0

                # 保存检查点
                if global_step % args.save_steps == 0 and rank == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(save_path, "pytorch_model.bin"))
                    print(f"Checkpoint saved to {save_path}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    # 最终保存
    if rank == 0:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'step': global_step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_path, "pytorch_model.bin"))
        print(f"\nTraining completed! Final model saved to {save_path}")

    # 清理
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="3D Parallel Training Demo")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练参数
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")

    # 并行参数
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3])

    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./output")

    # 训练框架
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1) 

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 选择训练方式
    if args.use_deepspeed:
        train_with_deepspeed(args)
    else:
        train_with_pytorch_ddp(args)


if __name__ == "__main__":
    main()
