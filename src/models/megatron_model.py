"""
megatron_model.py - Megatron风格模型，支持TP和PP（完全修复版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import math
from deepspeed.pipe import PipelineModule, LayerSpec

class SimpleTransformer(nn.Module):
    """简化的 Transformer 模型，用于单卡测试"""
    
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=512,
        num_heads=8,
        num_layers=8,
        ffn_hidden_size=2048,
        max_seq_len=128
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = self._get_positional_encoding(max_seq_len, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_hidden_size,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _get_positional_encoding(self, seq_len, hidden_size):
        """生成位置编码"""
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: 不使用（为了兼容 API）
            labels: (batch_size, seq_len)
        Returns:
            loss 或 logits
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入层
        x = self.embedding(input_ids)  # (batch_size, seq_len, hidden_size)
        
        # 添加位置编码
        pe = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pe
        
        # Transformer 解码层（不使用 mask，简化版本）
        for layer in self.layers:
            # 注意：TransformerDecoderLayer 需要 memory 参数
            # 这里我们使用自注意力，所以 memory = x
            x = layer(x, memory=x)
        
        # 层归一化
        x = self.norm(x)
        
        # 语言模型头（预测下一个 token）
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # 计算损失
        if labels is not None:
            # Flatten 以计算交叉熵
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),  # (batch_size * seq_len, vocab_size)
                labels.reshape(-1)  # (batch_size * seq_len,)
            )
            return {"loss":loss}
        
        return logits


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


class EmbeddingPipe(nn.Module):
    """
    Pipeline的第一层：输入 (input_ids, attention_mask)，输出 (hidden_states, attention_mask)
    """
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_len,
        hidden_dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.embedding_dropout = nn.Dropout(hidden_dropout)

    def forward(self, args):
        # DeepSpeed Pipeline传入的 args 通常是一个 tuple
        # 在第一层， args 来自 DataLoader，格式由你的 Dataset 决定
        # 假设格式为 (input_ids, attention_mask)
        input_ids, attention_mask = args
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        # 处理 Mask (转换格式)
        if attention_mask is not None:
            # 扩展 mask 维度以适应 Transformer: (batch, 1, 1, seq_len)
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 使用 fp16/bf16 兼容的大负数
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            else:
                extended_attention_mask = attention_mask
        else:
            extended_attention_mask = None
            
        extended_attention_mask = extended_attention_mask.clone().detach()
        extended_attention_mask.requires_grad_(True) 

        # 必须返回 tuple 以便传给下一层
        return hidden_states, extended_attention_mask


class TransformerLayerPipe(nn.Module):
    """
    中间层包装器：接收 (hidden, mask)，计算后返回 (hidden, mask)
    """
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        ffn_hidden_size,
        attention_dropout,
        hidden_dropout,
        layernorm_epsilon=1e-5,
        tp_group=None
    ):
        super().__init__()
        # 复用你原有的 ParallelTransformerLayer
        self.layer = ParallelTransformerLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            layernorm_epsilon=layernorm_epsilon,
            tp_group=tp_group
        )

    def forward(self, args):
        hidden_states, attention_mask = args
        # 调用原来的层
        output = self.layer(hidden_states, attention_mask)
        # 必须把 mask 继续往下传，因为后面的层也需要
        return output, attention_mask


class FinalLayerPipe(nn.Module):
    """
    Pipeline的最后一层：接收 (hidden, mask)，输出 logits
    """
    def __init__(self, hidden_size, vocab_size, layernorm_epsilon=1e-5):
        super().__init__()
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, args):
        hidden_states, _ = args # 最后一层不需要 mask 了
        
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Pipeline 的最后一层通常只返回模型的输出 (logits)
        # Loss 计算由 deepspeed 在外部通过 loss_fn 处理
        return logits
    
    
def get_megatron_pipeline_model(model_args, tp_group=None, num_stages=1):
    """
    构建 DeepSpeed PipelineModule
    """
    
    # 1. 定义 Embedding Spec
    specs = [
        LayerSpec(
            EmbeddingPipe,
            vocab_size=model_args.vocab_size,
            hidden_size=model_args.hidden_size,
            max_seq_len=model_args.max_seq_len,
            hidden_dropout=model_args.hidden_dropout
        )
    ]
    
    # 2. 定义 Transformer Layers Specs
    if model_args.ffn_hidden_size is None:
        model_args.ffn_hidden_size = 4 * model_args.hidden_size

    for _ in range(model_args.num_layers):
        specs.append(
            LayerSpec(
                TransformerLayerPipe,
                hidden_size=model_args.hidden_size,
                num_attention_heads=model_args.num_heads,
                ffn_hidden_size=model_args.ffn_hidden_size,
                attention_dropout=model_args.attention_dropout,
                hidden_dropout=model_args.hidden_dropout,
                tp_group=tp_group
            )
        )
        
    # 3. 定义 Final Layer Spec
    specs.append(
        LayerSpec(
            FinalLayerPipe,
            hidden_size=model_args.hidden_size,
            vocab_size=model_args.vocab_size,
        )
    )
    
    # 4. 实例化 PipelineModule
    # DeepSpeed 会自动根据 num_stages 将 specs 列表切分到不同 GPU
    
    def causal_lm_loss_fn(outputs, labels):
        """
        outputs: 模型最后一层的输出 (logits)
        labels: 数据集中的标签
        """
        logits = outputs
        vocab_size = logits.size(-1)
        
        # Shift logits and labels for causal LM task
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        return loss



    model = PipelineModule(
        layers=specs,
        loss_fn=causal_lm_loss_fn,
        num_stages=num_stages, # 对应 JSON 中的 pipeline.num_stages
        partition_method='type:TransformerLayerPipe', # 按照 Transformer 层进行负载均衡切分
        activation_checkpoint_interval=0 # 默认为0，如果显存不够可设为1
    )
    
    return model
