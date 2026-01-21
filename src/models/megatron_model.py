"""
Megatron-LM风格的模型并行实现
包含高级张量并行和通信优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import math


# ============= 通信原语 =============

def copy_to_tensor_model_parallel_region(input_: torch.Tensor):
    """在前向传播中复制，在反向传播中all-reduce"""
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor):
    """在前向传播中all-reduce，在反向传播中复制"""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor):
    """在前向传播中scatter，在反向传播中gather"""
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor):
    """在前向传播中gather，在反向传播中scatter"""
    return _GatherFromModelParallelRegion.apply(input_)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """前向复制，反向all-reduce"""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            dist.all_reduce(grad_output)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """前向all-reduce，反向复制"""

    @staticmethod
    def forward(ctx, input_):
        if dist.is_initialized():
            dist.all_reduce(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """前向scatter，反向gather"""

    @staticmethod
    def forward(ctx, input_):
        if not dist.is_initialized():
            return input_

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 按最后一维切分
        dim_size = input_.size(-1)
        assert dim_size % world_size == 0
        per_partition_size = dim_size // world_size

        # 切分
        output = input_[..., rank * per_partition_size:(rank + 1) * per_partition_size]
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_initialized():
            return grad_output

        world_size = dist.get_world_size()

        # Gather梯度
        grad_list = [torch.zeros_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_list, grad_output)

        # 拼接
        grad_input = torch.cat(grad_list, dim=-1)
        return grad_input


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """前向gather，反向scatter"""

    @staticmethod
    def forward(ctx, input_):
        if not dist.is_initialized():
            return input_

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ctx.rank = rank

        # All-gather
        output_list = [torch.zeros_like(input_) for _ in range(world_size)]
        dist.all_gather(output_list, input_)

        # 拼接
        output = torch.cat(output_list, dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_initialized():
            return grad_output

        world_size = dist.get_world_size()
        rank = ctx.rank

        # 切分梯度
        dim_size = grad_output.size(-1)
        per_partition_size = dim_size // world_size

        grad_input = grad_output[..., rank * per_partition_size:(rank + 1) * per_partition_size]
        return grad_input.contiguous()


# ============= Megatron并行层 =============

class ColumnParallelLinear(nn.Module):
    """
    列并行线性层（Megatron风格）

    权重矩阵: [output_size, input_size]
    列切分: 每个rank持有 [output_size/world_size, input_size]

    前向: Y = XA^T, 其中A按列切分
    输出: 每个rank产生部分输出
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        init_method: str = 'xavier_uniform',
        stride: int = 1
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # 获取并行配置
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # 计算每个partition的输出大小
        assert output_size % world_size == 0, \
            f"output_size ({output_size}) must be divisible by world_size ({world_size})"

        self.output_size_per_partition = output_size // world_size

        # 权重参数（只保存部分列）
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )
        else:
            self.register_parameter('bias', None)

        # 初始化
        self._initialize_weights(init_method)

    def _initialize_weights(self, method: str):
        """权重初始化"""
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight)
        elif method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init method: {method}")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_: [batch, seq_len, input_size]

        Returns:
            output: [batch, seq_len, output_size_per_partition]
                    或 [batch, seq_len, output_size] (如果gather_output=True)
        """
        # 在进入张量并行区域前，复制输入（反向时会all-reduce梯度）
        input_parallel = copy_to_tensor_model_parallel_region(input_)

        # 本地矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # 如果需要gather输出
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output


class RowParallelLinear(nn.Module):
    """
    行并行线性层（Megatron风格）

    权重矩阵: [output_size, input_size]
    行切分: 每个rank持有 [output_size, input_size/world_size]

    前向: Y = XA^T, 其中A按行切分
    输入: 每个rank接收部分输入（已被切分）
    输出: all-reduce后得到完整输出
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: str = 'xavier_uniform'
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        # 获取并行配置
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # 计算每个partition的输入大小
        assert input_size % world_size == 0, \
            f"input_size ({input_size}) must be divisible by world_size ({world_size})"

        self.input_size_per_partition = input_size // world_size

        # 权重参数（只保存部分行）
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)

        # 初始化
        self._initialize_weights(init_method)

    def _initialize_weights(self, method: str):
        """权重初始化"""
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight)
        elif method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init method: {method}")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_: [batch, seq_len, input_size_per_partition]
                    或 [batch, seq_len, input_size] (如果input_is_parallel=False)

        Returns:
            output: [batch, seq_len, output_size]
        """
        # 如果输入不是并行的，需要先scatter
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        # 本地矩阵乘法（不加bias）
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce得到完整输出
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        # 最后加bias（只在一个rank上）
        if self.bias is not None:
            output = output + self.bias

        return output


class ParallelMLP(nn.Module):
    """
    并行MLP（FFN）层

    结构:
    input -> ColumnParallel(up) -> activation -> RowParallel(down) -> output

    第一层列并行: 切分输出维度
    第二层行并行: 切分输入维度（与第一层输出对齐）
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = 'gelu',
        bias: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # 第一层: 列并行（不gather输出）
        self.dense_h_to_4h = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=ffn_hidden_size,
            bias=bias,
            gather_output=False  # 保持切分状态
        )

        # 激活函数
        if activation == 'gelu':
            self.activation_func = F.gelu
        elif activation == 'relu':
            self.activation_func = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 第二层: 行并行（输入已经是并行的）
        self.dense_4h_to_h = RowParallelLinear(
            input_size=ffn_hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True  # 输入已是并行切分的
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # 第一层: hidden_size -> ffn_hidden_size (并行切分)
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        # 激活
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # 第二层: ffn_hidden_size -> hidden_size (all-reduce)
        output = self.dense_4h_to_h(intermediate_parallel)

        # Dropout
        output = self.dropout(output)

        return output


class ParallelAttention(nn.Module):
    """
    并行多头注意力

    策略:
    - 注意力头按TP维度切分
    - QKV投影使用列并行（切分注意力头）
    - 输出投影使用行并行
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # 获取并行配置
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        assert num_attention_heads % world_size == 0, \
            f"num_attention_heads ({num_attention_heads}) must be divisible by world_size ({world_size})"

        self.num_attention_heads_per_partition = num_attention_heads // world_size
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads

        # QKV投影: 列并行（切分注意力头维度）
        self.query_key_value = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * hidden_size,
            bias=bias,
            gather_output=False
        )

        # 输出投影: 行并行
        self.dense = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True
        )

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        self.scale = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, 1, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape

        # QKV投影 [batch, seq_len, 3 * hidden_size_per_partition]
        mixed_x_layer = self.query_key_value(hidden_states)

        # 重塑为 [batch, seq_len, num_heads_per_partition, 3 * head_dim]
        new_x_shape = (
            batch_size,
            seq_length,
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head
        )
        mixed_x_layer = mixed_x_layer.view(*new_x_shape)

        # 分离Q, K, V
        query_layer, key_layer, value_layer = mixed_x_layer.chunk(3, dim=-1)

        # 转置: [batch, num_heads_per_partition, seq_len, head_dim]
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        # 注意力分数: [batch, num_heads_per_partition, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale

        # 应用mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # 应用attention: [batch, num_heads_per_partition, seq_len, head_dim]
        context_layer = torch.matmul(attention_probs, value_layer)

        # 转置回来: [batch, seq_len, num_heads_per_partition, head_dim]
        context_layer = context_layer.transpose(1, 2).contiguous()

        # 重塑: [batch, seq_len, hidden_size_per_partition]
        new_context_shape = (batch_size, seq_length, -1)
        context_layer = context_layer.view(*new_context_shape)

        # 输出投影（行并行，会all-reduce）
        output = self.dense(context_layer)
        output = self.hidden_dropout(output)

        return output


class ParallelTransformerLayer(nn.Module):
    """
    并行Transformer层
    包含ParallelAttention和ParallelMLP
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5
    ):
        super().__init__()

        # Layer Norm 1
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Attention
        self.attention = ParallelAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout
        )

        # Layer Norm 2
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout=hidden_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, 1, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Attention + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        # MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        output = residual + mlp_output

        return output


class MegatronLLM(nn.Module):
    """
    Megatron风格的大语言模型
    支持张量并行和流水线并行
    """

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
        layernorm_epsilon: float = 1e-5
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # Embedding层（不并行，或使用vocab并行）
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.embedding_dropout = nn.Dropout(hidden_dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            ParallelTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=ffn_hidden_size,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layernorm_epsilon=layernorm_epsilon
            )
            for _ in range(num_layers)
        ])

        # 最终Layer Norm
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        # LM Head（可以与embedding权重共享）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len]

        Returns:
            dict with 'loss', 'logits', 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        # 处理attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 最终Layer Norm
        hidden_states = self.final_layernorm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        # 计算损失
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

    def get_num_params(self):
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())
