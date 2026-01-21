"""
train_3d_parallel.py
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from loguru import logger

from src.models.megatron_model3d import MegatronLLM
from src.parallel.pipeline_parallel import InterleaveEngine, PipelineEngine, GPipeEngine
from src.checkpoint import CheckpointManager, BestCheckpointManager
from src.dataset import SimpleTextDataset



def setup_distributed_environment():
    """初始化分布式环境"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"Setting up distributed: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    torch.cuda.set_device(local_rank)
    logger.info(f"Rank {rank} initialized on device {local_rank}")
    
    return local_rank, rank, world_size


def create_parallel_groups(world_size, tp_size, pp_size):
    """创建TP/PP/DP通信组"""
    rank = dist.get_rank()
    dp_size = world_size // (tp_size * pp_size)
    
    # 映射: rank = dp_idx * (pp_size * tp_size) + pp_idx * tp_size + tp_idx
    dp_rank = rank // (pp_size * tp_size)
    pp_rank = (rank // tp_size) % pp_size
    tp_rank = rank % tp_size
    
    logger.info(f"Rank {rank}: DP={dp_rank}, PP={pp_rank}, TP={tp_rank}")
    
    tp_group = None
    pp_group = None
    dp_group = None

    # 创建TP组 (同一个DP、同一个PP下的所有TP rank)
    if tp_size > 1:
        for dp in range(dp_size):
            for pp in range(pp_size):
                ranks = [dp * (pp_size * tp_size) + pp * tp_size + tp for tp in range(tp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = group
                    logger.info(f"Rank {rank} joined TP group: {ranks}")
    
    # 创建DP组 (同一个PP、同一个TP下的所有DP rank)
    if dp_size > 1:
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = [dp * (pp_size * tp_size) + pp * tp_size + tp for dp in range(dp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    dp_group = group
                    logger.info(f"Rank {rank} joined DP group: {ranks}")
    
    logger.info(f"Rank {rank}: PP communication will use world group (absolute rank)")
    
    return tp_group, pp_group, dp_group, tp_rank, pp_rank, dp_rank


def create_model_with_pp(args, pp_rank, tp_group):
    """创建带PP切分的模型"""
    model = MegatronLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        ffn_hidden_size=args.ffn_hidden_size,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        tp_group=tp_group
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
            del model.final_layernorm
            del model.lm_head
    
    return model


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
            train_loader = tqdm(train_loader, desc=f"Training")
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--ffn_hidden_size", type=int, default=None)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--pipeline_schedule", type=str, default="1f1b", choices=["1f1b", "gpipe"])
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    args = parser.parse_args()
    
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size
    
    # 初始化分布式
    local_rank, rank, world_size = setup_distributed_environment()
    
    # 计算DP size
    args.dp_size = world_size // (args.tp_size * args.pp_size)
    
    # 创建通信组
    tp_group, pp_group, dp_group, tp_rank, pp_rank, dp_rank = create_parallel_groups(
        world_size, args.tp_size, args.pp_size
    )
    
    if rank == 0:
        logger.info(f"3D并行配置: DP={args.dp_size}, TP={args.tp_size}, PP={args.pp_size}")
    
    # 创建模型
    model = create_model_with_pp(args, pp_rank, tp_group)
    
    # 创建数据集
    train_dataset = SimpleTextDataset(
        num_samples=args.num_samples,
        seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.dp_size,
        rank=dp_rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )
    
    # 创建训练器
    trainer = Trainer3D(
        model, args, local_rank, rank, world_size,
        tp_group, pp_group, dp_group,
        tp_rank, pp_rank, dp_rank
    )
    
    # 训练
    trainer.train(train_loader)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()