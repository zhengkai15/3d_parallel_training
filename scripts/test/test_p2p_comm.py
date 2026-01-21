"""
test_p2p_comm.py - 测试P2P通信是否正常工作
"""
import os
import torch
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_p2p():
    """测试P2P通信"""
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    logger.info(f"Rank {rank}/{world_size} on device {device}")
    
    # 等待所有进程初始化完毕
    dist.barrier()
    
    # 测试1: 简单的点对点通信
    if world_size >= 2:
        logger.info(f"[Rank {rank}] Test 1: Simple P2P communication")
        
        if rank == 0:
            # Rank 0 发送到 Rank 1
            tensor = torch.ones(10, dtype=torch.float32, device=device)
            logger.info(f"[Rank {rank}] Sending to rank 1")
            dist.send(tensor, dst=1)
            logger.info(f"[Rank {rank}] Send complete")
            
        elif rank == 1:
            # Rank 1 从 Rank 0 接收
            tensor = torch.zeros(10, dtype=torch.float32, device=device)
            logger.info(f"[Rank {rank}] Receiving from rank 0")
            dist.recv(tensor, src=0)
            logger.info(f"[Rank {rank}] Receive complete, tensor sum: {tensor.sum()}")
        
        dist.barrier()
    
    # 测试2: 异步P2P通信
    if world_size >= 4:
        logger.info(f"[Rank {rank}] Test 2: Async P2P communication")
        
        tensor = torch.ones(10, dtype=torch.float32, device=device) * rank
        
        if rank == 0:
            # 0 -> 1, 0 -> 2, 0 -> 3
            logger.info(f"[Rank {rank}] Async sending to ranks 1, 2, 3")
            reqs = []
            for dst in [1, 2, 3]:
                req = dist.isend(tensor.clone(), dst=dst)
                reqs.append(req)
            
            for req in reqs:
                req.wait()
            logger.info(f"[Rank {rank}] All sends complete")
            
        elif rank > 0:
            # 1, 2, 3 从 0 接收
            recv_tensor = torch.zeros(10, dtype=torch.float32, device=device)
            logger.info(f"[Rank {rank}] Receiving from rank 0")
            dist.recv(recv_tensor, src=0)
            logger.info(f"[Rank {rank}] Receive complete, tensor sum: {recv_tensor.sum()}")
        
        dist.barrier()
    
    # 测试3: Pipeline通信模式（stage 0->1->2->3）
    if world_size == 4:
        logger.info(f"[Rank {rank}] Test 3: Pipeline communication pattern")
        
        for step in range(3):
            logger.info(f"[Rank {rank}] Pipeline step {step}")
            
            # Forward pass
            tensor = torch.ones(10, dtype=torch.float32, device=device) * rank
            
            if rank < 3:
                # 发送到下一个rank
                logger.info(f"[Rank {rank}] Sending to rank {rank+1}")
                dist.send(tensor, dst=rank+1)
                logger.info(f"[Rank {rank}] Send complete")
            
            if rank > 0:
                # 从前一个rank接收
                recv_tensor = torch.zeros(10, dtype=torch.float32, device=device)
                logger.info(f"[Rank {rank}] Receiving from rank {rank-1}")
                dist.recv(recv_tensor, src=rank-1)
                logger.info(f"[Rank {rank}] Receive complete, tensor: {recv_tensor}")
            
            dist.barrier()
        
        logger.info(f"[Rank {rank}] Pipeline test complete")
    
    dist.barrier()
    logger.info(f"[Rank {rank}] All tests passed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    test_p2p()