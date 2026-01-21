"""
Unified 3D Parallel Training Framework
支持多种训练策略: DDP, DeepSpeed, Pipeline Parallel, Tensor Parallel
"""
import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from loguru import logger
import numpy as np
from typing import Optional, Dict, Any
import deepspeed

from src.models.megatron_model import MegatronLLM
from src.dataset import SimpleTextDataset
from src.trainers.comm import *

# ============================================================================
# 工具函数
# ============================================================================

def setup_distributed():
    """初始化分布式环境"""
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def create_parallel_groups(world_size, tp_size, pp_size):
    """创建TP/PP/DP通信组"""
    rank = dist.get_rank()
    dp_size = world_size // (tp_size * pp_size)
    
    dp_rank = rank // (pp_size * tp_size)
    pp_rank = (rank // tp_size) % pp_size
    tp_rank = rank % tp_size
    
    tp_group, pp_group, dp_group = None, None, None
    
    if tp_size > 1:
        for dp in range(dp_size):
            for pp in range(pp_size):
                ranks = [dp * (pp_size * tp_size) + pp * tp_size + tp for tp in range(tp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = group
    
    if dp_size > 1:
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = [dp * (pp_size * tp_size) + pp * tp_size + tp for dp in range(dp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    dp_group = group
    
    return tp_group, pp_group, dp_group, tp_rank, pp_rank, dp_rank


def create_model(args):
    """创建模型"""
    return MegatronLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        ffn_hidden_size=args.ffn_hidden_size or 4 * args.hidden_size,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout
    )

def create_model_with_pp(args, pp_rank, tp_group):
    """创建带PP切分的模型（传递TP group）"""
    model = MegatronLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        ffn_hidden_size=args.ffn_hidden_size,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        tp_group=tp_group  # ✅ 传递TP group给模型
    )
    
    if args.pp_size > 1:
        # 计算每个stage的layers
        layers_per_stage = args.num_layers // args.pp_size
        start_layer = pp_rank * layers_per_stage
        end_layer = (pp_rank + 1) * layers_per_stage if pp_rank < args.pp_size - 1 else args.num_layers
        
        logger.info(f"PP rank {pp_rank}: layers {start_layer}~{end_layer}")
        
        # 只保留当前stage的layers
        model.layers = nn.ModuleList(list(model.layers)[start_layer:end_layer])
        
        # 第一阶段保留embedding，其他阶段删除
        if pp_rank != 0:
            del model.token_embedding
            del model.position_embedding
            del model.embedding_dropout
        
        # 最后阶段保留输出层，其他阶段删除
        if pp_rank != args.pp_size - 1:
            if hasattr(model, 'final_layernorm'):
                del model.final_layernorm
            if hasattr(model, 'lm_head'):
                del model.lm_head
    
    return model

# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--ffn_hidden_size", type=int, default=None)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    
    # 数据参数
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    
    # 训练参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Adam参数
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Trainer选择
    parser.add_argument("--trainer", type=str, default="simple")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 并行参数
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--pipeline_schedule", type=str, default="1f1b", choices=["1f1b", "gpipe"])
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3])
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    local_rank, rank, world_size = setup_distributed()
    
    args.dp_size = world_size // (args.tp_size * args.pp_size)
    
    if rank == 0:
        logger.info(f"3D Parallel Config: DP={args.dp_size}, TP={args.tp_size}, PP={args.pp_size}")
        logger.info(f"Using {args.trainer} trainer")
    
    # 创建模型
    model = create_model(args)
    
    # 创建数据集
    train_dataset = SimpleTextDataset(
        num_samples=args.num_samples,
        seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )
    
    # 选择Trainer
    if args.trainer == "simple":
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler
        )
        trainer = SimpleTrainer(model, args, local_rank, rank, world_size)
    
    elif args.trainer == "3dtrainer":
        tp_group, pp_group, dp_group, tp_rank, pp_rank, dp_rank = create_parallel_groups(
            world_size, args.tp_size, args.pp_size
        )
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.dp_size, rank=dp_rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler
        )
        
        model = create_model_with_pp(args, pp_rank, tp_group)
        
        trainer = Trainer3D(
        model, args, local_rank, rank, world_size,
        tp_group, pp_group, dp_group,
        tp_rank, pp_rank, dp_rank
        )
        
    elif args.trainer == "megatron_trainer":  
        MegatronTrainer(
            model=model,
            args=args,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size
        )
    
    elif args.trainer == "deepspeed":
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True
        )
        trainer = DeepSpeedTrainer(model, args, local_rank, rank, world_size)
    
    else:
        raise ValueError(f"Unknown trainer: {args.trainer}")
    
    # 开始训练
    trainer.train(train_loader)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()