"""
Megatron-LM风格的训练脚本
完整实现Megatron的训练策略和优化技巧
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import time
import json
from datetime import datetime
from typing import Optional
import numpy as np
from src.models.megatron_model import MegatronLLM
from src.dataset import SimpleTextDataset

def setup_distributed():
    """初始化分布式环境"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def get_learning_rate_scheduler(optimizer, args):
    """
    Megatron学习率调度策略
    - Warmup阶段: 线性增长到max_lr
    - Decay阶段: 余弦衰减到min_lr
    """
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Warmup: 从0线性增长到1
            return float(current_step) / float(max(1, args.warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - args.warmup_steps) / \
                      float(max(1, args.max_steps - args.warmup_steps))
            return max(args.min_lr / args.learning_rate,
                      0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


class MegatronTrainer:
    """Megatron训练器"""

    def __init__(self, model, args, local_rank, rank, world_size):
        self.model = model
        self.args = args
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = get_learning_rate_scheduler(self.optimizer, args)

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

        # 统计
        self.global_step = 0
        self.tokens_trained = 0
        self.start_time = time.time()

        # 创建输出目录
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            self.log_file = open(os.path.join(args.output_dir, 'train.log'), 'w')

    def _create_optimizer(self):
        """
        创建优化器（Megatron优化策略）
        - 对不同参数组使用不同的weight decay
        - LayerNorm和bias不使用weight decay
        """
        # 分离参数组
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_eps
        )

        return optimizer

    def train_step(self, batch):
        """单步训练"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # 混合精度前向传播
        with torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=torch.float16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']

        return loss

    def train(self, train_loader):
        """训练主循环"""
        self.model.train()

        if self.rank == 0:
            self._print_training_config()

        epoch = 0
        running_loss = 0.0
        log_loss = 0.0

        self.optimizer.zero_grad()

        while self.global_step < self.args.max_steps:
            epoch += 1

            if self.rank == 0:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch}")
                print(f"{'='*80}")

            for step, batch in enumerate(train_loader):
                step_start_time = time.time()

                # 训练步
                loss = self.train_step(batch)

                # 梯度缩放和反向传播
                loss_scaled = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss_scaled).backward()

                running_loss += loss.item()
                log_loss += loss.item()

                # 梯度累积
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # 更新统计
                    self.global_step += 1
                    self.tokens_trained += (
                        self.args.batch_size *
                        self.world_size *
                        self.args.max_seq_len *
                        self.args.gradient_accumulation_steps
                    )

                    step_time = time.time() - step_start_time

                    # 日志
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_training_stats(
                            loss=log_loss / self.args.logging_steps,
                            grad_norm=grad_norm,
                            step_time=step_time
                        )
                        log_loss = 0.0

                    # 保存检查点
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()

                    # 评估
                    if self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    if self.global_step >= self.args.max_steps:
                        break

            if self.global_step >= self.args.max_steps:
                break

        # 最终保存
        if self.rank == 0:
            self.save_checkpoint(is_final=True)
            print(f"\n{'='*80}")
            print("Training Completed!")
            print(f"{'='*80}")
            self.log_file.close()

    def _print_training_config(self):
        """打印训练配置"""
        print(f"\n{'='*80}")
        print("Megatron Training Configuration")
        print(f"{'='*80}")
        print(f"Model:")
        print(f"  Vocab size: {self.args.vocab_size}")
        print(f"  Hidden size: {self.args.hidden_size}")
        print(f"  Num layers: {self.args.num_layers}")
        print(f"  Num heads: {self.args.num_heads}")
        print(f"  FFN hidden size: {self.args.ffn_hidden_size}")
        print(f"  Max sequence length: {self.args.max_seq_len}")
        print(f"  Total parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"\nTraining:")
        print(f"  World size: {self.world_size}")
        print(f"  Batch size per GPU: {self.args.batch_size}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Global batch size: {self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}")
        print(f"  Max steps: {self.args.max_steps}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Min learning rate: {self.args.min_lr}")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Weight decay: {self.args.weight_decay}")
        print(f"  Gradient clipping: {self.args.max_grad_norm}")
        print(f"  Mixed precision: {self.args.fp16}")
        print(f"\nLogging:")
        print(f"  Logging steps: {self.args.logging_steps}")
        print(f"  Save steps: {self.args.save_steps}")
        print(f"  Output dir: {self.args.output_dir}")
        print(f"{'='*80}\n")

    def _log_training_stats(self, loss, grad_norm, step_time):
        """记录训练统计"""
        lr = self.scheduler.get_last_lr()[0]
        elapsed_time = time.time() - self.start_time
        samples_per_sec = (self.args.batch_size * self.world_size) / step_time
        tokens_per_sec = samples_per_sec * self.args.max_seq_len

        # 计算剩余时间
        steps_remaining = self.args.max_steps - self.global_step
        estimated_time = steps_remaining * step_time
        hours = int(estimated_time // 3600)
        minutes = int((estimated_time % 3600) // 60)

        log_str = (
            f"Step {self.global_step}/{self.args.max_steps} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"GradNorm: {grad_norm:.2f} | "
            f"Time: {step_time:.2f}s | "
            f"Samples/s: {samples_per_sec:.1f} | "
            f"Tokens/s: {tokens_per_sec:.0f} | "
            f"ETA: {hours}h{minutes}m"
        )

        if self.rank == 0:
            print(log_str)
            self.log_file.write(log_str + '\n')
            self.log_file.flush()

            # 保存指标到JSON
            metrics = {
                'step': self.global_step,
                'loss': loss,
                'learning_rate': lr,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'samples_per_second': samples_per_sec,
                'tokens_per_second': tokens_per_sec,
                'elapsed_time': elapsed_time,
                'tokens_trained': self.tokens_trained
            }

            metrics_file = os.path.join(self.args.output_dir, 'metrics.jsonl')
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

    def save_checkpoint(self, is_final=False):
        """保存检查点"""
        if self.rank != 0:
            return

        save_dir = os.path.join(
            self.args.output_dir,
            "final_model" if is_final else f"checkpoint-{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.args.fp16 else None,
            'tokens_trained': self.tokens_trained,
            'args': vars(self.args)
        }

        torch.save(checkpoint, os.path.join(save_dir, 'pytorch_model.bin'))

        # 保存配置
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=2)

        print(f"Checkpoint saved to {save_dir}")

    def evaluate(self):
        """评估（简化版）"""
        # TODO: 实现完整的评估逻辑
        pass


def main():
    parser = argparse.ArgumentParser(description="Megatron-LM Training")

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
    parser.add_argument("--output_dir", type=str, default="./output_megatron")

    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    # FFN hidden size默认值
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    # 初始化分布式
    local_rank, rank, world_size = setup_distributed()

    # 设置随机种子
    seed = 1234 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # 创建模型
    model = MegatronLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        ffn_hidden_size=args.ffn_hidden_size,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout
    ).cuda()

    # 创建数据集
    train_dataset = SimpleTextDataset(
        num_samples=args.num_samples,
        seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 创建训练器
    trainer = MegatronTrainer(
        model=model,
        args=args,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size
    )

    # 恢复检查点（如果有）
    if args.resume_from_checkpoint:
        # TODO: 实现检查点恢复
        pass

    # 开始训练
    trainer.train(train_loader)

    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
