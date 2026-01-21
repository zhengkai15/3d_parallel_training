"""
megatron_model.py - Megatron风格模型，支持TP和PP（完全修复版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import math


# ============= TP通信原语 =============

def copy_to_tensor_model_parallel_region(input_: torch.Tensor, tp_group=None):
    """在前向传播中复制，在反向传播中all-reduce（仅在TP group内）"""
    return _CopyToModelParallelRegion.apply(input_, tp_group)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor, tp_group=None):
    """在前向传播中all-reduce（仅在TP group内），在反向传播中复制"""
    return _ReduceFromModelParallelRegion.apply(input_, tp_group)


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, tp_group=None):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and ctx.tp_group is not None:
            dist.all_reduce(grad_output, group=ctx.tp_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, tp_group=None):
        ctx.tp_group = tp_group
        if dist.is_initialized() and tp_group is not None:
            dist.all_reduce(input_, group=tp_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ============= TP并行层 =============

class ColumnParallelLinear(nn.Module):
    """列并行线性层"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        init_method: str = 'xavier_uniform',
        tp_group = None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.tp_group = tp_group

        world_size = dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1

        assert output_size % world_size == 0
        self.output_size_per_partition = output_size // world_size

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights(init_method)

    def _initialize_weights(self, method: str):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight)
        elif method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_model_parallel_region(input_, tp_group=self.tp_group)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output and dist.is_initialized() and self.tp_group is not None:
            world_size = dist.get_world_size(self.tp_group)
            if world_size > 1:
                output_list = [torch.zeros_like(output_parallel) for _ in range(world_size)]
                dist.all_gather(output_list, output_parallel, group=self.tp_group)
                output = torch.cat(output_list, dim=-1)
            else:
                output = output_parallel
        else:
            output = output_parallel

        return output


class RowParallelLinear(nn.Module):
    """行并行线性层"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: str = 'xavier_uniform',
        tp_group = None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group

        world_size = dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1

        assert input_size % world_size == 0
        self.input_size_per_partition = input_size // world_size

        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights(init_method)

    def _initialize_weights(self, method: str):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight)
        elif method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and dist.is_initialized() and self.tp_group is not None:
            world_size = dist.get_world_size(self.tp_group)
            rank = dist.get_rank(self.tp_group)
            if world_size > 1:
                dim_size = input_.size(-1)
                per_partition_size = dim_size // world_size
                input_parallel = input_[..., rank * per_partition_size:(rank + 1) * per_partition_size]
            else:
                input_parallel = input_
        else:
            input_parallel = input_

        output_parallel = F.linear(input_parallel, self.weight)
        output = reduce_from_tensor_model_parallel_region(output_parallel, tp_group=self.tp_group)

        if self.bias is not None:
            output = output + self.bias

        return output


class ParallelMLP(nn.Module):
    """并行MLP"""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = 'gelu',
        bias: bool = True,
        dropout: float = 0.1,
        tp_group = None
    ):
        super().__init__()

        self.dense_h_to_4h = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=ffn_hidden_size,
            bias=bias,
            gather_output=False,
            tp_group=tp_group
        )

        if activation == 'gelu':
            self.activation_func = F.gelu
        elif activation == 'relu':
            self.activation_func = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dense_4h_to_h = RowParallelLinear(
            input_size=ffn_hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            tp_group=tp_group
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        return output


class ParallelAttention(nn.Module):
    """并行多头注意力"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        bias: bool = True,
        tp_group = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.tp_group = tp_group

        world_size = dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1

        assert num_attention_heads % world_size == 0

        self.num_attention_heads_per_partition = num_attention_heads // world_size
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads

        self.query_key_value = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * hidden_size,
            bias=bias,
            gather_output=False,
            tp_group=tp_group
        )

        self.dense = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            tp_group=tp_group
        )

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.scale = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        mixed_x_layer = self.query_key_value(hidden_states)

        new_x_shape = (
            batch_size,
            seq_length,
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head
        )
        mixed_x_layer = mixed_x_layer.view(*new_x_shape)

        query_layer, key_layer, value_layer = mixed_x_layer.chunk(3, dim=-1)

        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()

        new_context_shape = (batch_size, seq_length, -1)
        context_layer = context_layer.view(*new_context_shape)

        output = self.dense(context_layer)
        output = self.hidden_dropout(output)

        return output


class ParallelTransformerLayer(nn.Module):
    """并行Transformer层"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        tp_group = None
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.attention = ParallelAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            tp_group=tp_group
        )

        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout=hidden_dropout,
            tp_group=tp_group
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        output = residual + mlp_output

        return output


class MegatronLLM(nn.Module):
    """Megatron风格的LLM - 支持Pipeline并行"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        max_seq_len: int,
        ffn_hidden_size: Optional[int] = None,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        tp_group = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tp_group = tp_group

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # 仅第一个stage有embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.embedding_dropout = nn.Dropout(hidden_dropout)

        self.layers = nn.ModuleList([
            ParallelTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=ffn_hidden_size,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layernorm_epsilon=layernorm_epsilon,
                tp_group=tp_group
            )
            for _ in range(num_layers)
        ])

        # 仅最后一个stage有输出层
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_first_stage: bool = True,
        is_last_stage: bool = True
    ):
        """
        Args:
            hidden_states: 
                - 第一个stage: input_ids (batch_size, seq_len)
                - 其他stage: hidden_states (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) 或 None
            labels: (batch_size, seq_len) 或 None
            is_first_stage: 是否是第一个pipeline stage
            is_last_stage: 是否是最后一个pipeline stage
        """
        
        # ========== 处理第一个stage的embedding ==========
        if is_first_stage:
            input_ids = hidden_states  # 第一个stage接收input_ids
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            token_embeds = self.token_embedding(input_ids)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds
            hidden_states = self.embedding_dropout(hidden_states)
        else:
            # 其他stage直接接收hidden_states，已经是(batch_size, seq_len, hidden_size)
            batch_size, seq_len, _ = hidden_states.shape

        # ========== 处理attention mask ==========
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # ========== 执行Transformer层 ==========
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # ========== 处理最后一个stage的输出 ==========
        if is_last_stage:
            hidden_states = self.final_layernorm(hidden_states)
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )

            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states
            }
        else:
            # 中间stage仅返回hidden_states供下一stage使用
            return {
                'hidden_states': hidden_states,
                'logits': None,
                'loss': None
            }

    def get_num_params(self):
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())