"""
checkpoint.py - 模型保存和加载（支持3D并行）
"""
import os
import torch
import torch.distributed as dist
from pathlib import Path
import json
from loguru import logger
from typing import Dict, Optional


class CheckpointManager:
    """管理3D并行模型的保存和加载"""
    
    def __init__(
        self,
        output_dir: str,
        tp_rank: int,
        pp_rank: int,
        dp_rank: int,
        tp_size: int,
        pp_size: int,
        dp_size: int,
        rank: int,
        world_size: int
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.rank = rank
        self.world_size = world_size
        
        self.is_master = (rank == 0)
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: int,
        args: Dict,
        avg_loss: float
    ):
        """保存checkpoint（支持PP/TP切分）"""
        
        logger.info(f"[Rank {self.rank}] Saving checkpoint at step {step}")
        
        # 创建step目录
        step_dir = self.output_dir / f"checkpoint-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== 保存模型状态 ==========
        # 对于PP并行，每个rank的model.layers不同
        # 对于TP并行，weight参数被分割
        
        model_state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'avg_loss': avg_loss,
            'parallel_config': {
                'tp_rank': self.tp_rank,
                'pp_rank': self.pp_rank,
                'dp_rank': self.dp_rank,
                'tp_size': self.tp_size,
                'pp_size': self.pp_size,
                'dp_size': self.dp_size,
                'world_size': self.world_size,
            }
        }
        
        # 每个rank保存自己的状态（对于TP/PP并行）
        rank_ckpt_path = step_dir / f"rank_{self.rank}_state.pt"
        torch.save(model_state, rank_ckpt_path)
        logger.info(f"[Rank {self.rank}] Saved rank checkpoint to {rank_ckpt_path}")
        
        # master rank保存training参数
        if self.is_master:
            args_path = step_dir / "args.json"
            with open(args_path, 'w') as f:
                # 将args中的非序列化对象移除
                args_to_save = {k: v for k, v in vars(args).items() 
                               if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                json.dump(args_to_save, f, indent=2)
            logger.info(f"[Rank 0] Saved training args to {args_path}")
            
            # 保存meta信息
            meta_path = step_dir / "meta.json"
            meta = {
                'step': step,
                'avg_loss': avg_loss,
                'parallel_config': {
                    'tp_size': self.tp_size,
                    'pp_size': self.pp_size,
                    'dp_size': self.dp_size,
                    'world_size': self.world_size,
                }
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f"[Rank 0] Saved metadata to {meta_path}")
        
        # 同步所有rank
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"[Rank {self.rank}] Checkpoint saved successfully")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        checkpoint_dir: str
    ) -> Dict:
        """加载checkpoint"""
        
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return {}
        
        logger.info(f"[Rank {self.rank}] Loading checkpoint from {checkpoint_dir}")
        
        # 加载rank特定的状态
        rank_ckpt_path = checkpoint_path / f"rank_{self.rank}_state.pt"
        if not rank_ckpt_path.exists():
            logger.error(f"Rank checkpoint not found: {rank_ckpt_path}")
            return {}
        
        device = next(model.parameters()).device
        checkpoint = torch.load(rank_ckpt_path, map_location=device)
        
        # 加载模型
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"[Rank {self.rank}] Loaded model state")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to load model state: {e}")
        
        # 加载优化器
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"[Rank {self.rank}] Loaded optimizer state")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to load optimizer state: {e}")
        
        # 加载scheduler
        if scheduler and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"[Rank {self.rank}] Loaded scheduler state")
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Failed to load scheduler state: {e}")
        
        # 同步所有rank
        if dist.is_initialized():
            dist.barrier()
        
        # master rank加载meta信息
        meta = {}
        if self.is_master:
            meta_path = checkpoint_path / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                logger.info(f"[Rank 0] Loaded metadata: step={meta.get('step')}, loss={meta.get('avg_loss')}")
        
        # 广播meta信息
        if dist.is_initialized():
            meta_list = [meta] if self.is_master else [None]
            dist.broadcast_object_list(meta_list, src=0)
            meta = meta_list[0]
        
        logger.info(f"[Rank {self.rank}] Checkpoint loaded successfully")
        return meta

    def save_final_model(
        self,
        model: torch.nn.Module,
        args: Dict
    ):
        """保存最终模型（去除优化器状态以减小文件大小）"""
        
        logger.info(f"[Rank {self.rank}] Saving final model")
        
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存model_state_dict
        model_path = final_dir / f"rank_{self.rank}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"[Rank {self.rank}] Saved model to {model_path}")
        
        # master rank保存配置
        if self.is_master:
            config_path = final_dir / "config.json"
            config = {
                'vocab_size': args.vocab_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'max_seq_len': args.max_seq_len,
                'ffn_hidden_size': args.ffn_hidden_size,
                'parallel_config': {
                    'tp_size': self.tp_size,
                    'pp_size': self.pp_size,
                    'dp_size': self.dp_size,
                }
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"[Rank 0] Saved model config to {config_path}")
        
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"[Rank {self.rank}] Final model saved successfully")

    @staticmethod
    def load_distributed_model(model: torch.nn.Module, checkpoint_dir: str, rank: int, world_size: int):
        """加载分布式模型（用于推理时）"""
        
        checkpoint_path = Path(checkpoint_dir)
        device = next(model.parameters()).device
        
        # 尝试加载当前rank的模型
        rank_path = checkpoint_path / f"rank_{rank}_model.pt"
        if rank_path.exists():
            checkpoint = torch.load(rank_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            logger.info(f"[Rank {rank}] Loaded model from {rank_path}")
        else:
            logger.warning(f"[Rank {rank}] Model file not found: {rank_path}")
        
        return model


class BestCheckpointManager:
    """管理最佳checkpoint（基于validation loss）"""
    
    def __init__(self, output_dir: str, keep_best_k: int = 3):
        self.output_dir = Path(output_dir)
        self.keep_best_k = keep_best_k
        self.best_checkpoints = []  # List of (loss, step) tuples
        
    def update(self, loss: float, step: int, checkpoint_dir: str):
        """如果是更好的checkpoint，则更新"""
        
        is_best = False
        if len(self.best_checkpoints) < self.keep_best_k:
            is_best = True
        elif loss < max(self.best_checkpoints, key=lambda x: x[0])[0]:
            is_best = True
            # 删除最差的checkpoint
            worst_loss, worst_step = max(self.best_checkpoints, key=lambda x: x[0])
            self.best_checkpoints.remove((worst_loss, worst_step))
            worst_dir = self.output_dir / f"checkpoint-{worst_step}"
            if worst_dir.exists():
                import shutil
                shutil.rmtree(worst_dir)
                logger.info(f"Removed checkpoint-{worst_step}")
        
        if is_best:
            self.best_checkpoints.append((loss, step))
            self.best_checkpoints.sort(key=lambda x: x[0])
            logger.info(f"New best checkpoint at step {step} with loss {loss:.4f}")
            return True
        
        return False