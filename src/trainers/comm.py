"""
Unified 3D Parallel Training Framework
支持多种训练策略: DDP, DeepSpeed, Pipeline Parallel, Tensor Parallel
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
from typing import Optional, Dict, Any
import deepspeed

from src.parallel.pipeline_parallel import InterleaveEngine, PipelineEngine, GPipeEngine
from src.checkpoint import CheckpointManager, BestCheckpointManager


# ============================================================================
# 基础Trainer类
# ============================================================================

class BaseTrainer(ABC):
    """所有Trainer的基类"""
    
    def __init__(self, model, args, local_rank, rank, world_size):
        self.model = model
        self.args = args
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        
        self.global_step = 0
        self.tokens_trained = 0
        self.start_time = time.time()
        
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            self.log_file = open(os.path.join(args.output_dir, 'train.log'), 'w')
    
    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        """单步训练，返回loss"""
        pass
    
    @abstractmethod
    def train(self, train_loader):
        """训练主循环"""
        pass
    
    def _log_training_stats(self, loss, grad_norm, step_time):
        """记录训练统计"""
        if self.rank != 0:
            return
        
        elapsed_time = time.time() - self.start_time
        samples_per_sec = (self.args.batch_size * self.world_size) / step_time
        tokens_per_sec = samples_per_sec * self.args.max_seq_len
        
        steps_remaining = self.args.max_steps - self.global_step
        estimated_time = steps_remaining * step_time
        hours = int(estimated_time // 3600)
        minutes = int((estimated_time % 3600) // 60)
        
        log_str = (
            f"Step {self.global_step}/{self.args.max_steps} | "
            f"Loss: {loss:.4f} | "
            f"GradNorm: {grad_norm:.2f} | "
            f"Time: {step_time:.2f}s | "
            f"Samples/s: {samples_per_sec:.1f} | "
            f"Tokens/s: {tokens_per_sec:.0f} | "
            f"ETA: {hours}h{minutes}m"
        )
        
        print(log_str)
        self.log_file.write(log_str + '\n')
        self.log_file.flush()


# ============================================================================
# 具体Trainer实现
# ============================================================================

class SimpleTrainer(BaseTrainer):
    """标准DDP训练器"""
    
    def __init__(self, model, args, local_rank, rank, world_size):
        super().__init__(model, args, local_rank, rank, world_size)
        
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[local_rank])
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
        if rank == 0:
            logger.info("SimpleTrainer initialized (DDP)")
    
    def _create_optimizer(self):
        """创建优化器，LayerNorm和bias不使用weight decay"""
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
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_eps
        )
    
    def _create_scheduler(self):
        """Cosine decay with warmup"""
        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return float(step) / float(max(1, self.args.warmup_steps))
            progress = (step - self.args.warmup_steps) / max(1, self.args.max_steps - self.args.warmup_steps)
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch) -> torch.Tensor:
        """单步训练"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
        
        return loss
    
    def train(self, train_loader):
        """训练主循环"""
        self.model.train()
        optimizer = self.optimizer
        
        if self.rank == 0:
            logger.info("Starting training with SimpleTrainer...")
        
        self.global_step = 0
        running_loss = 0.0
        
        while self.global_step < self.args.max_steps:
            for step, batch in enumerate(train_loader):
                step_start = time.time()
                
                loss = self.train_step(batch)
                loss_scaled = loss / self.args.gradient_accumulation_steps
                
                self.scaler.scale(loss_scaled).backward()
                running_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    optimizer.zero_grad()
                    
                    self.global_step += 1
                    step_time = time.time() - step_start
                    
                    if self.global_step % self.args.logging_steps == 0:
                        avg_loss = running_loss / self.args.logging_steps
                        self._log_training_stats(avg_loss, grad_norm, step_time)
                        running_loss = 0.0
                    
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()
                    
                    if self.global_step >= self.args.max_steps:
                        break
            
            if self.global_step >= self.args.max_steps:
                break
        
        self._save_checkpoint(is_final=True)
        if self.rank == 0:
            logger.info("Training completed!")
    
    def _save_checkpoint(self, is_final=False):
        """保存检查点"""
        if self.rank != 0:
            return
        
        save_dir = os.path.join(
            self.args.output_dir,
            "final_model" if is_final else f"checkpoint-{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': vars(self.args)
        }, os.path.join(save_dir, 'pytorch_model.bin'))
        
        logger.info(f"Checkpoint saved to {save_dir}")



class DeepSpeedTrainer(BaseTrainer):
    """DeepSpeed训练器 (支持ZeRO优化)"""
    
    def __init__(self, model, args, local_rank, rank, world_size):
        super().__init__(model, args, local_rank, rank, world_size)
        
        self.local_rank = local_rank
        deepspeed.init_distributed()
        
        ds_config = self._get_ds_config()
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        
        if rank == 0:
            logger.info(f"DeepSpeedTrainer initialized (ZeRO stage={args.zero_stage})")
    
    def _get_ds_config(self) -> Dict[str, Any]:
        """获取DeepSpeed配置"""
        if self.args.deepspeed_config and os.path.exists(self.args.deepspeed_config):
            with open(self.args.deepspeed_config) as f:
                return json.load(f)
        
        return {
            "train_batch_size": self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "steps_per_print": self.args.logging_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.args.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.args.learning_rate,
                    "warmup_num_steps": self.args.warmup_steps,
                    "total_num_steps": self.args.max_steps
                }
            },
            "fp16": {"enabled": self.args.fp16},
            "zero_optimization": {"stage": self.args.zero_stage}
        }
    
    def train_step(self, batch) -> torch.Tensor:
        """单步训练"""
        input_ids = batch['input_ids'].to(self.model_engine.device)
        attention_mask = batch['attention_mask'].to(self.model_engine.device)
        labels = batch['labels'].to(self.model_engine.device)
        if self.args.fp16:
            attention_mask = attention_mask.half()
        outputs = self.model_engine(input_ids, attention_mask, labels=labels)
        return outputs['loss']
    
    def train(self, train_loader):
        """训练主循环"""
        self.model_engine.train()
        
        if self.rank == 0:
            logger.info("Starting training with DeepSpeedTrainer...")
        
        self.global_step = 0
        running_loss = 0.0
        
        while self.global_step < self.args.max_steps:
            for step, batch in enumerate(train_loader):
                step_start = time.time()
                
                loss = self.train_step(batch)
                self.model_engine.backward(loss)
                self.model_engine.step()
                
                self.global_step += 1
                running_loss += loss.item()
                step_time = time.time() - step_start
                
                if self.global_step % self.args.logging_steps == 0 and self.rank == 0:
                    avg_loss = running_loss / self.args.logging_steps
                    self._log_training_stats(avg_loss, 0, step_time)
                    running_loss = 0.0
                
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint()
                
                if self.global_step >= self.args.max_steps:
                    break
            
            if self.global_step >= self.args.max_steps:
                break
        
        if self.rank == 0:
            logger.info("Training completed!")
    
    def _save_checkpoint(self, is_final=False):
        """保存检查点"""
        tag = "final_model" if is_final else f"checkpoint-{self.global_step}"
        self.model_engine.save_checkpoint(self.args.output_dir, tag=tag)
        if self.rank == 0:
            logger.info(f"Checkpoint saved with tag: {tag}")



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
                input_ids,
                attention_mask,
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

class Trainer3D:
    """3D并行训练器"""
    
    def __init__(self, model, args, local_rank, rank, world_size,
                 tp_group, pp_group, dp_group, tp_rank, pp_rank, dp_rank):
        self.model = model
        self.args = args
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        
        self.tp_group = tp_group
        self.pp_group = pp_group
        self.dp_group = dp_group
        
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        
        self.device = torch.device(f'cuda:{local_rank}')
        self.model = self.model.to(self.device)
        
        # 确保model有hidden_size属性
        if not hasattr(self.model, 'hidden_size'):
            self.model.hidden_size = args.hidden_size
        
        logger.info(f"Rank {self.rank}: Model moved to {self.device}")
        
        # 创建pipeline engine
        self.pipeline_engine = None
        if args.pp_size > 1:
            topology_config = {
                "tp_size": args.tp_size,
                "pp_size": args.pp_size,
                "dp_size": args.dp_size,
                "world_size": world_size
            }
            
            if args.pipeline_schedule == '1f1b':
                EngineClass = InterleaveEngine
            elif args.pipeline_schedule == 'gpipe':
                EngineClass = GPipeEngine
            else:
                EngineClass = InterleaveEngine
            
            self.pipeline_engine = EngineClass(
                model=self.model,
                num_stages=args.pp_size,
                num_microbatches=args.num_microbatches,
                stage_id=pp_rank,
                device=self.device,
                loss_fn=nn.CrossEntropyLoss(),
                topology_config=topology_config
            )
            logger.info(f"Rank {self.rank}: Pipeline engine ({args.pipeline_schedule}) created")
        
        # 创建checkpoint管理器
        self.checkpoint_manager = CheckpointManager(
            output_dir=args.output_dir,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            dp_size=args.dp_size,
            rank=rank,
            world_size=world_size
        )
        self.best_ckpt_manager = BestCheckpointManager(args.output_dir, keep_best_k=3)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps
        )
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    def train_step(self, batch):
        """训练一步"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if self.pipeline_engine is not None:
            # 使用pipeline
            if isinstance(self.pipeline_engine, GPipeEngine):
                loss = self.pipeline_engine.train_batch_gpipe(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels if self.pp_rank == self.args.pp_size - 1 else None
                )
            else:
                # 1F1B
                loss = self.pipeline_engine.train_batch_1f1b(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels if self.pp_rank == self.args.pp_size - 1 else None
                )
        else:
            # 标准训练
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = self.model(
                    hidden_states=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    is_first_stage=True,
                    is_last_stage=True
                )
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
        
        return loss
    
    def train(self, train_loader):
        """训练主循环"""
        self.model.train()
        global_step = 0
        total_loss = 0.0
        
        if self.rank == 0:
            logger.info("开始训练...")
            from tqdm import tqdm
            train_loader = tqdm(train_loader, total=self.args.max_steps, desc="Training")
        
        while global_step < self.args.max_steps:
            for step, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # 训练步
                loss = self.train_step(batch)
                
                # DP梯度同步
                if self.args.dp_size > 1 and self.dp_group is not None:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)
                
                # 梯度裁剪和参数更新
                if self.args.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                if self.args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                
                # 累计loss（用于计算平均值）
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                total_loss += loss_val
                
                global_step += 1
                
                # 日志
                if global_step % self.args.logging_steps == 0 and self.rank == 0:
                    avg_loss = total_loss / self.args.logging_steps
                    if self.pp_rank == self.args.pp_size - 1 or self.args.pp_size == 1:
                        logger.info(f"Step {global_step}/{self.args.max_steps} | Loss: {avg_loss:.4f}")
                    total_loss = 0.0
                
                # 保存checkpoint
                if global_step % self.args.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=global_step,
                        args=self.args,
                        avg_loss=loss_val
                    )
                    if self.rank == 0:
                        logger.info(f"Checkpoint saved at step {global_step}")
                
                if global_step >= self.args.max_steps:
                    break
            
            if global_step >= self.args.max_steps:
                break
        
        # 保存最终模型
        self.checkpoint_manager.save_final_model(self.model, self.args)
        
        if self.rank == 0:
            logger.info("训练完成！")