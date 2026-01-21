"""
simple_pp_test.py - 简化版PP训练测试（修复梯度问题）
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [Rank %(rankno)d] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class LogRankFilter(logging.Filter):
    def filter(self, record):
        record.rankno = int(os.environ.get('RANK', 0))
        return True

logger.addFilter(LogRankFilter())


class SimpleModel(nn.Module):
    """简单模型用于测试"""
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x


def test_simple_pp():
    """测试简单的PP"""
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    logger.info(f"Rank {rank}/{world_size} on device {device}, starting test")
    
    dist.barrier()
    
    # 参数
    hidden_size = 64
    seq_len = 10
    batch_size = 4
    num_stages = 2  # 与world_size一致用于简单测试
    
    if world_size != num_stages:
        logger.error(f"Please use {num_stages} GPUs for this test")
        return
    
    stage_id = rank
    is_first_stage = (rank == 0)
    is_last_stage = (rank == num_stages - 1)
    
    logger.info(f"Rank {rank}: stage_id={stage_id}, is_first={is_first_stage}, is_last={is_last_stage}")
    
    # 创建模型
    model = SimpleModel(hidden_size=hidden_size, num_layers=1).to(device)
    model.hidden_size = hidden_size
    
    logger.info(f"Rank {rank}: Model created")
    
    dist.barrier()
    
    # === 测试前向传播 ===
    logger.info(f"Rank {rank}: Starting forward pass test")
    
    if is_first_stage:
        # Stage 0: 生成输入（需要梯度用于反向）
        input_data = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
        logger.info(f"Rank {rank}: Generated input, shape={input_data.shape}")
        
        # 前向计算
        output = model(input_data)
        logger.info(f"Rank {rank}: Computed output, shape={output.shape}")
        
        # 保留梯度（非叶子张量需要显式保留）
        output.retain_grad()
        
        # 发送到下一阶段
        logger.info(f"Rank {rank}: Sending to rank {rank+1}")
        dist.send(output.detach(), dst=rank+1)
        logger.info(f"Rank {rank}: Send complete")
        
    else:
        # Stage 1: 接收前一阶段的输出
        recv_tensor = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        logger.info(f"Rank {rank}: Waiting to receive from rank {rank-1}")
        dist.recv(recv_tensor, src=rank-1)
        logger.info(f"Rank {rank}: Received, shape={recv_tensor.shape}")
        
        # 需要梯度用于反向
        recv_tensor.requires_grad = True
        
        # 前向计算
        output = model(recv_tensor)
        logger.info(f"Rank {rank}: Computed output, shape={output.shape}")
        
        # 保留梯度
        output.retain_grad()
    
    dist.barrier()
    logger.info(f"Rank {rank}: Forward pass test complete")
    
    # === 测试反向传播 ===
    logger.info(f"Rank {rank}: Starting backward pass test")
    
    if is_last_stage:
        # Stage 1: 计算loss并反向
        logger.info(f"Rank {rank}: output shape={output.shape}, requires_grad={output.requires_grad}")
        
        loss = output.mean()
        logger.info(f"Rank {rank}: Loss={loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        logger.info(f"Rank {rank}: Executed loss.backward()")
        
        # 获取output梯度（现在应该存在了）
        if output.grad is not None:
            grad = output.grad
            logger.info(f"Rank {rank}: Output grad shape={grad.shape}, grad_norm={grad.norm().item():.6f}")
            
            # 发送梯度到前一阶段
            logger.info(f"Rank {rank}: Sending grad to rank {rank-1}")
            dist.send(grad.clone(), dst=rank-1)
            logger.info(f"Rank {rank}: Grad send complete")
        else:
            logger.error(f"Rank {rank}: Output grad is None after backward!")
            logger.error(f"Rank {rank}: output.requires_grad={output.requires_grad}")
            logger.error(f"Rank {rank}: is_leaf={output.is_leaf}")
    
    else:
        # Stage 0: 接收梯度并反向
        logger.info(f"Rank {rank}: Waiting to receive grad from rank {rank+1}")
        
        grad = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        dist.recv(grad, src=rank+1)
        logger.info(f"Rank {rank}: Received grad, shape={grad.shape}, grad_norm={grad.norm().item():.6f}")
        
        # 使用接收到的梯度进行反向
        if output.requires_grad:
            logger.info(f"Rank {rank}: output requires_grad={output.requires_grad}")
            logger.info(f"Rank {rank}: Executing output.backward(grad)")
            output.backward(grad)
            logger.info(f"Rank {rank}: Backward complete")
            
            # 检查input梯度
            if hasattr(output, '_prev_input_data'):
                logger.info(f"Rank {rank}: Input grad computed")
        else:
            logger.error(f"Rank {rank}: Output doesn't require grad!")
    
    dist.barrier()
    logger.info(f"Rank {rank}: Backward pass test complete")
    
    logger.info(f"Rank {rank}: All tests passed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    test_simple_pp()