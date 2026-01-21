"""
3D并行训练 - 模型定义
支持张量并行的Transformer模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TensorParallelLinear(nn.Module):
    """张量并行的线性层 - 列切分"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # 每个GPU只持有部分列
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.out_features_per_partition)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: [batch, seq_len, in_features]
        # 输出: [batch, seq_len, out_features_per_partition]
        output = F.linear(x, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    """张量并行的线性层 - 行切分"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # 每个GPU只持有部分行
        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: [batch, seq_len, in_features_per_partition]
        # 本地计算
        output = F.linear(x, self.weight)

        # All-reduce跨GPU合并结果
        if self.world_size > 1:
            torch.distributed.all_reduce(output)

        return output


class TensorParallelAttention(nn.Module):
    """张量并行的多头注意力"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank

        # 每个TP rank持有部分注意力头
        self.num_heads_per_partition = num_heads // tp_world_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V投影 - 列并行
        self.qkv_proj = TensorParallelLinear(
            hidden_size,
            3 * hidden_size,
            tp_world_size,
            tp_rank
        )

        # 输出投影 - 行并行
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_world_size,
            tp_rank
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV投影
        qkv = self.qkv_proj(hidden_states)

        # 分割Q, K, V
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            self.num_heads_per_partition,
            3 * self.head_dim
        )
        q, k, v = qkv.chunk(3, dim=-1)

        # 转置以便进行批量矩阵乘法
        q = q.transpose(1, 2)  # [batch, num_heads_per_partition, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用注意力mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)

        # 重新整形
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size,
            seq_len,
            self.num_heads_per_partition * self.head_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """支持张量并行的Transformer块"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()

        # 注意力层
        self.attention = TensorParallelAttention(
            hidden_size, num_heads, tp_world_size, tp_rank, dropout
        )

        # FFN层 - 第一层列并行
        self.ffn1 = TensorParallelLinear(
            hidden_size,
            ffn_hidden_size,
            tp_world_size,
            tp_rank
        )

        # FFN层 - 第二层行并行
        self.ffn2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            tp_world_size,
            tp_rank
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 注意力 + 残差
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # FFN + 残差
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LLMModel(nn.Module):
    """
    简化的大语言模型
    支持3D并行: DDP + Pipeline + Tensor Parallelism
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        pp_stage: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.pp_stage = pp_stage

        ffn_hidden_size = 4 * hidden_size

        # Token嵌入 (stage 0)
        if pp_stage is None or pp_stage == 0:
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
            self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
            self.dropout = nn.Dropout(dropout)

        # Transformer层 (按pipeline stage分配)
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # 如果使用pipeline并行，只创建属于当前stage的层
            if pp_stage is None or self._layer_belongs_to_stage(layer_idx, pp_stage):
                layer = TransformerBlock(
                    hidden_size,
                    num_heads,
                    ffn_hidden_size,
                    tp_world_size,
                    tp_rank,
                    dropout
                )
                self.layers.append(layer)

        # 输出层 (最后一个stage)
        if pp_stage is None or self._is_last_stage(pp_stage):
            self.ln_f = nn.LayerNorm(hidden_size)
            # LM head可以与embedding共享权重
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _layer_belongs_to_stage(self, layer_idx: int, stage: int) -> bool:
        """判断某层是否属于某个pipeline stage"""
        # 简单平均分配策略
        layers_per_stage = self.num_layers // 4  # 假设4个stage
        return layer_idx // layers_per_stage == stage

    def _is_last_stage(self, stage: int) -> bool:
        """判断是否是最后一个stage"""
        return stage == 3  # 假设4个stage

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Stage 0: 嵌入层
        if self.pp_stage is None or self.pp_stage == 0:
            # Token embedding
            hidden_states = self.token_embedding(input_ids)

            # Position embedding
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(position_ids)

            hidden_states = hidden_states + position_embeds
            hidden_states = self.dropout(hidden_states)
        else:
            # 如果是后续stage，hidden_states会作为输入传入
            hidden_states = input_ids

        # 处理attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 最后一个stage: 输出层
        if self.pp_stage is None or self._is_last_stage(self.pp_stage):
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

            # 计算损失
            loss = None
            if labels is not None:
                # 展平用于计算损失
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
            # 中间stage只返回hidden_states
            return {'hidden_states': hidden_states}

    def get_num_params(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters())
