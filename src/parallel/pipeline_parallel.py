"""
pipeline_parallel.py - 完全修复版本（正确处理hidden_states传递）
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Dict, List
import time
from loguru import logger

class InterleaveEngine:
    """1F1B Pipeline - 修复版（正确处理hidden_states）"""
    
    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_microbatches: int,
        stage_id: int,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        topology_config: Optional[Dict] = None
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stage_id = stage_id
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

        # 获取拓扑配置
        self.tp_size = topology_config.get('tp_size', 1) if topology_config else 1
        self.pp_size = topology_config.get('pp_size', 1) if topology_config else 1
        self.dp_size = topology_config.get('dp_size', 1) if topology_config else 1
        self.world_size = topology_config.get('world_size', 1) if topology_config else 1
        
        self.current_rank = dist.get_rank()
        
        # rank映射: rank = dp_idx * (pp_size * tp_size) + pp_idx * tp_size + tp_idx
        self.dp_rank = self.current_rank // (self.pp_size * self.tp_size)
        self.pp_rank = (self.current_rank // self.tp_size) % self.pp_size
        self.tp_rank = self.current_rank % self.tp_size
        
        logger.info(f"Rank {self.current_rank}: DP={self.dp_rank}, PP={self.pp_rank}, TP={self.tp_rank}, "
                   f"Stage={stage_id}, Device={device}")
        
        # PP通信rank（同一DP+TP组内）
        if not self.is_first_stage:
            self.prev_stage_rank = self.current_rank - self.tp_size
        else:
            self.prev_stage_rank = None
        
        if not self.is_last_stage:
            self.next_stage_rank = self.current_rank + self.tp_size
        else:
            self.next_stage_rank = None

        logger.info(f"Rank {self.current_rank}: prev_stage_rank={self.prev_stage_rank}, "
                   f"next_stage_rank={self.next_stage_rank}")

        # 前向/反向缓存
        self.fwd_cache: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []
        self.send_reqs: List[dist.Work] = []
        
    def _forward_step(
        self, 
        input_tensor: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        recv_shape: Tuple,
        microbatch_idx: int
    ) -> torch.Tensor:
        """执行一个microbatch的前向传播"""
        
        # logger.debug(f"[Rank {self.current_rank}] Forward step {microbatch_idx} starting")
        
        # ========== 接收前一阶段的数据 ==========
        if not self.is_first_stage:
            recv_tensor = torch.empty(recv_shape, dtype=torch.float32, device=self.device)
            # logger.debug(f"[Rank {self.current_rank}] Waiting to recv from rank {self.prev_stage_rank}")
            dist.recv(recv_tensor, src=self.prev_stage_rank)
            # logger.debug(f"[Rank {self.current_rank}] Received from rank {self.prev_stage_rank}")
            recv_tensor.requires_grad = True
            input_tensor = recv_tensor
        
        # ========== 前向计算（传递is_first_stage和is_last_stage） ==========
        # logger.debug(f"[Rank {self.current_rank}] Computing forward")
        with torch.enable_grad():
            output = self.model(
                hidden_states=input_tensor,
                attention_mask=attention_mask,
                labels=None,  # 前向不需要labels
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage
            )
        
        # 提取hidden_states（中间stage是hidden_states，最后stage是logits）
        if isinstance(output, dict):
            if self.is_last_stage:
                hidden_states = output.get('logits')
            else:
                hidden_states = output.get('hidden_states')
        else:
            hidden_states = output
        
        # ========== 中间stage需要retain_grad以用于反向 ==========
        if not self.is_last_stage:
            hidden_states.retain_grad()
        
        # ========== 发送到下一阶段 ==========
        if not self.is_last_stage:
            # logger.debug(f"[Rank {self.current_rank}] Sending to rank {self.next_stage_rank}")
            req = dist.isend(hidden_states.detach().contiguous(), dst=self.next_stage_rank)
            self.send_reqs.append(req)
            # logger.debug(f"[Rank {self.current_rank}] Sent to rank {self.next_stage_rank}")
        
        # 缓存用于反向
        self.fwd_cache.append((input_tensor, hidden_states))
        
        # logger.debug(f"[Rank {self.current_rank}] Forward step {microbatch_idx} complete")
        return hidden_states

    def _backward_step(
        self, 
        labels: Optional[torch.Tensor], 
        recv_shape: Tuple,
        microbatch_idx: int
    ) -> float:
        """执行一个microbatch的反向传播"""
        
        # logger.debug(f"[Rank {self.current_rank}] Backward step {microbatch_idx} starting")
        
        # 等待所有send完成
        for req in self.send_reqs:
            req.wait()
        self.send_reqs.clear()
        
        # 取出缓存
        if not self.fwd_cache:
            logger.error(f"[Rank {self.current_rank}] fwd_cache is empty!")
            return 0.0
            
        input_tensor, output_tensor = self.fwd_cache.pop(0)
        loss_val = 0.0

        # ========== 接收梯度 ==========
        if not self.is_last_stage:
            grad_output = torch.empty_like(output_tensor)
            # logger.debug(f"[Rank {self.current_rank}] Waiting to recv grad from rank {self.next_stage_rank}")
            dist.recv(grad_output, src=self.next_stage_rank)
            # logger.debug(f"[Rank {self.current_rank}] Received grad from rank {self.next_stage_rank}")
        else:
            grad_output = None

        # ========== 反向传播 ==========
        # logger.debug(f"[Rank {self.current_rank}] Computing backward")
        
        if self.is_last_stage:
            # 最后stage：计算loss
            logits = output_tensor
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss_val = loss.item()
            # logger.debug(f"[Rank {self.current_rank}] Loss: {loss_val}")
            loss.backward()
        else:
            # 中间stage：使用接收到的梯度
            if grad_output is not None:
                output_tensor.backward(grad_output)
        
        # ========== 发送输入梯度 ==========
        if not self.is_first_stage:
            if input_tensor.grad is not None:
                # logger.debug(f"[Rank {self.current_rank}] Sending grad to rank {self.prev_stage_rank}")
                req = dist.isend(input_tensor.grad.contiguous(), dst=self.prev_stage_rank)
                self.send_reqs.append(req)
                # logger.debug(f"[Rank {self.current_rank}] Sent grad to rank {self.prev_stage_rank}")
        
        # logger.debug(f"[Rank {self.current_rank}] Backward step {microbatch_idx} complete")
        return loss_val

    def train_batch_1f1b(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """1F1B调度的训练batch"""
        
        # logger.info(f"[Rank {self.current_rank}] train_batch_1f1b starting")
        
        self.fwd_cache.clear()
        self.send_reqs.clear()
        
        batch_size = input_ids.size(0)
        assert batch_size % self.num_microbatches == 0, \
            f"Batch size {batch_size} must be divisible by num_microbatches {self.num_microbatches}"
        
        micro_batch_size = batch_size // self.num_microbatches
        
        seq_len = input_ids.size(1)
        hidden_size = getattr(self.model, 'hidden_size', 768)
        
        # ========== 重要：recv_shape用于接收hidden_states ==========
        if self.is_first_stage:
            # 第一stage发送的是embedding后的hidden_states
            recv_shape = (micro_batch_size, seq_len, hidden_size)
        else:
            # 中间/最后stage接收的都是hidden_states
            recv_shape = (micro_batch_size, seq_len, hidden_size)
        
        losses = []
        
        # 切分数据为microbatches
        micro_inputs = torch.split(input_ids, micro_batch_size)
        micro_masks = torch.split(attention_mask, micro_batch_size) if attention_mask is not None else [None] * self.num_microbatches
        micro_labels = torch.split(labels, micro_batch_size) if labels is not None else [None] * self.num_microbatches

        # === 1F1B调度 ===
        warmup_steps = min(self.num_stages - self.stage_id - 1, self.num_microbatches)
        remaining_steps = self.num_microbatches - warmup_steps
        
        # logger.info(f"[Rank {self.current_rank}] 1F1B schedule: warmup={warmup_steps}, remaining={remaining_steps}")
        
        # Phase 1: Warmup (仅Forward)
        # logger.info(f"[Rank {self.current_rank}] Warmup phase starting")
        for i in range(warmup_steps):
            # logger.info(f"[Rank {self.current_rank}] Warmup forward {i}/{warmup_steps}")
            self._forward_step(
                micro_inputs[i] if self.is_first_stage else None,
                micro_masks[i] if self.is_first_stage else None,
                recv_shape,
                i
            )
            time.sleep(0.01)
        
        # logger.info(f"[Rank {self.current_rank}] Warmup complete")
        
        # Phase 2: 1F1B (Forward + Backward)
        # logger.info(f"[Rank {self.current_rank}] 1F1B phase starting")
        fwd_idx = warmup_steps
        bwd_idx = 0
        
        for step in range(remaining_steps):
            # logger.info(f"[Rank {self.current_rank}] 1F1B step {step}/{remaining_steps}")
            
            # Forward
            self._forward_step(
                micro_inputs[fwd_idx] if self.is_first_stage else None,
                micro_masks[fwd_idx] if self.is_first_stage else None,
                recv_shape,
                fwd_idx
            )
            fwd_idx += 1
            
            time.sleep(0.01)
            
            # Backward
            loss = self._backward_step(
                micro_labels[bwd_idx] if self.is_last_stage else None,
                recv_shape,
                bwd_idx
            )
            if self.is_last_stage:
                losses.append(loss)
            bwd_idx += 1
        
        # logger.info(f"[Rank {self.current_rank}] 1F1B phase complete")
        
        # Phase 3: Cooldown (仅Backward)
        # logger.info(f"[Rank {self.current_rank}] Cooldown phase starting")
        for i in range(warmup_steps):
            # logger.info(f"[Rank {self.current_rank}] Cooldown backward {i}/{warmup_steps}")
            loss = self._backward_step(
                micro_labels[bwd_idx] if self.is_last_stage else None,
                recv_shape,
                bwd_idx
            )
            if self.is_last_stage:
                losses.append(loss)
            bwd_idx += 1
            time.sleep(0.01)
        
        # logger.info(f"[Rank {self.current_rank}] Cooldown complete")
        
        # 等待所有send完成
        for req in self.send_reqs:
            req.wait()
        self.send_reqs.clear()
        
        # 返回平均loss
        if losses:
            avg_loss = sum(losses) / len(losses)
            return torch.tensor(avg_loss, device=self.device)
        
        return torch.tensor(0.0, device=self.device)


class PipelineEngine(InterleaveEngine):
    """Pipeline 并行引擎"""
    def train_batch(self, *args, **kwargs):
        return self.train_batch_1f1b(*args, **kwargs)
    


class GPipeEngine:
    """GPipe - 同步流水线并行（按顺序执行）"""
    
    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_microbatches: int,
        stage_id: int,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        topology_config: Optional[Dict] = None
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stage_id = stage_id
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

        # 获取拓扑配置
        self.tp_size = topology_config.get('tp_size', 1) if topology_config else 1
        self.pp_size = topology_config.get('pp_size', 1) if topology_config else 1
        self.dp_size = topology_config.get('dp_size', 1) if topology_config else 1
        self.world_size = topology_config.get('world_size', 1) if topology_config else 1
        
        self.current_rank = dist.get_rank()
        
        # rank映射: rank = dp_idx * (pp_size * tp_size) + pp_idx * tp_size + tp_idx
        self.dp_rank = self.current_rank // (self.pp_size * self.tp_size)
        self.pp_rank = (self.current_rank // self.tp_size) % self.pp_size
        self.tp_rank = self.current_rank % self.tp_size
        
        logger.info(f"Rank {self.current_rank}: DP={self.dp_rank}, PP={self.pp_rank}, TP={self.tp_rank}, "
                   f"Stage={stage_id}, Device={device}")
        
        # PP通信rank
        if not self.is_first_stage:
            self.prev_stage_rank = self.current_rank - self.tp_size
        else:
            self.prev_stage_rank = None
        
        if not self.is_last_stage:
            self.next_stage_rank = self.current_rank + self.tp_size
        else:
            self.next_stage_rank = None

        logger.info(f"Rank {self.current_rank}: prev_stage_rank={self.prev_stage_rank}, "
                   f"next_stage_rank={self.next_stage_rank}")

        # 缓存
        self.fwd_cache: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []
        self.send_reqs: List[dist.Work] = []
        
    def _forward_step(
        self, 
        input_tensor: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        recv_shape: Tuple,
        microbatch_idx: int
    ) -> torch.Tensor:
        """执行一个microbatch的前向传播"""
        
        # logger.debug(f"[Rank {self.current_rank}] Forward step {microbatch_idx} starting")
        
        # 接收前一阶段的数据
        if not self.is_first_stage:
            recv_tensor = torch.empty(recv_shape, dtype=torch.float32, device=self.device)
            # logger.debug(f"[Rank {self.current_rank}] Waiting to recv from rank {self.prev_stage_rank}")
            dist.recv(recv_tensor, src=self.prev_stage_rank)
            # logger.debug(f"[Rank {self.current_rank}] Received from rank {self.prev_stage_rank}")
            recv_tensor.requires_grad = True
            input_tensor = recv_tensor
        
        # 前向计算
        # logger.debug(f"[Rank {self.current_rank}] Computing forward")
        with torch.enable_grad():
            output = self.model(
                hidden_states=input_tensor,
                attention_mask=attention_mask,
                labels=None,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage
            )
        
        # 提取hidden_states
        if isinstance(output, dict):
            if self.is_last_stage:
                hidden_states = output.get('logits')
            else:
                hidden_states = output.get('hidden_states')
        else:
            hidden_states = output
        
        # 中间stage需要retain_grad
        if not self.is_last_stage:
            hidden_states.retain_grad()
        
        # 发送到下一阶段
        if not self.is_last_stage:
            # logger.debug(f"[Rank {self.current_rank}] Sending to rank {self.next_stage_rank}")
            req = dist.isend(hidden_states.detach().contiguous(), dst=self.next_stage_rank)
            self.send_reqs.append(req)
            # logger.debug(f"[Rank {self.current_rank}] Sent to rank {self.next_stage_rank}")
        
        # 缓存用于反向
        self.fwd_cache.append((input_tensor, hidden_states))
        
        # logger.debug(f"[Rank {self.current_rank}] Forward step {microbatch_idx} complete")
        return hidden_states

    def _backward_step(
        self, 
        labels: Optional[torch.Tensor], 
        recv_shape: Tuple,
        microbatch_idx: int
    ) -> float:
        """执行一个microbatch的反向传播"""
        
        # logger.debug(f"[Rank {self.current_rank}] Backward step {microbatch_idx} starting")
        
        # 等待所有send完成
        for req in self.send_reqs:
            req.wait()
        self.send_reqs.clear()
        
        # 取出缓存
        if not self.fwd_cache:
            logger.error(f"[Rank {self.current_rank}] fwd_cache is empty!")
            return 0.0
            
        input_tensor, output_tensor = self.fwd_cache.pop(0)
        loss_val = 0.0

        # 接收梯度
        if not self.is_last_stage:
            grad_output = torch.empty_like(output_tensor)
            # logger.debug(f"[Rank {self.current_rank}] Waiting to recv grad from rank {self.next_stage_rank}")
            dist.recv(grad_output, src=self.next_stage_rank)
            # logger.debug(f"[Rank {self.current_rank}] Received grad from rank {self.next_stage_rank}")
        else:
            grad_output = None

        # 反向传播
        # logger.debug(f"[Rank {self.current_rank}] Computing backward")
        
        if self.is_last_stage:
            # 最后stage：计算loss
            logits = output_tensor
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss_val = loss.item()
            # logger.debug(f"[Rank {self.current_rank}] Loss: {loss_val}")
            loss.backward()
        else:
            # 中间stage：使用接收到的梯度
            if grad_output is not None:
                output_tensor.backward(grad_output)
        
        # 发送输入梯度
        if not self.is_first_stage:
            if input_tensor.grad is not None:
                # logger.debug(f"[Rank {self.current_rank}] Sending grad to rank {self.prev_stage_rank}")
                req = dist.isend(input_tensor.grad.contiguous(), dst=self.prev_stage_rank)
                self.send_reqs.append(req)
                # logger.debug(f"[Rank {self.current_rank}] Sent grad to rank {self.prev_stage_rank}")
        
        # logger.debug(f"[Rank {self.current_rank}] Backward step {microbatch_idx} complete")
        return loss_val

    def train_batch_gpipe(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """GPipe调度 - 顺序执行所有microbatches的前向，再顺序执行反向"""
        
        # logger.info(f"[Rank {self.current_rank}] train_batch_gpipe starting")
        
        self.fwd_cache.clear()
        self.send_reqs.clear()
        
        batch_size = input_ids.size(0)
        assert batch_size % self.num_microbatches == 0, \
            f"Batch size {batch_size} must be divisible by num_microbatches {self.num_microbatches}"
        
        micro_batch_size = batch_size // self.num_microbatches
        
        seq_len = input_ids.size(1)
        hidden_size = getattr(self.model, 'hidden_size', 768)
        recv_shape = (micro_batch_size, seq_len, hidden_size)
        
        losses = []
        
        # 切分数据为microbatches
        micro_inputs = torch.split(input_ids, micro_batch_size)
        micro_masks = torch.split(attention_mask, micro_batch_size) if attention_mask is not None else [None] * self.num_microbatches
        micro_labels = torch.split(labels, micro_batch_size) if labels is not None else [None] * self.num_microbatches

        # === GPipe: 所有Forward + 所有Backward ===
        
        # Phase 1: 所有Microbatches的Forward
        # logger.info(f"[Rank {self.current_rank}] GPipe forward phase starting")
        for i in range(self.num_microbatches):
            # logger.info(f"[Rank {self.current_rank}] Forward {i}/{self.num_microbatches}")
            self._forward_step(
                micro_inputs[i] if self.is_first_stage else None,
                micro_masks[i] if self.is_first_stage else None,
                recv_shape,
                i
            )
        
        # logger.info(f"[Rank {self.current_rank}] Forward phase complete")
        
        # 等待所有forward send完成
        for req in self.send_reqs:
            req.wait()
        self.send_reqs.clear()
        
        # Phase 2: 所有Microbatches的Backward
        # logger.info(f"[Rank {self.current_rank}] GPipe backward phase starting")
        for i in range(self.num_microbatches):
            # logger.info(f"[Rank {self.current_rank}] Backward {i}/{self.num_microbatches}")
            loss = self._backward_step(
                micro_labels[i] if self.is_last_stage else None,
                recv_shape,
                i
            )
            if self.is_last_stage:
                losses.append(loss)
        
        # logger.info(f"[Rank {self.current_rank}] Backward phase complete")
        
        # 等待所有backward send完成
        for req in self.send_reqs:
            req.wait()
        self.send_reqs.clear()
        
        # 返回平均loss
        if losses:
            avg_loss = sum(losses) / len(losses)
            return torch.tensor(avg_loss, device=self.device)
        
        return torch.tensor(0.0, device=self.device)