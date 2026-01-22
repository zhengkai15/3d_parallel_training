#!/bin/bash

# Megatron-LM风格训练启动脚本
# 支持单节点和多节点训练
# 使用方法: bash scripts/run_megatron.sh

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

# 使用配置文件或命令行参数
CONFIG_FILE=${CONFIG_FILE:-"configs/models/small.yaml"}

# GPU配置
NUM_GPUS=${NUM_GPUS:-4}

# 节点配置（多节点训练）
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# 模型配置（如果不使用配置文件）
VOCAB_SIZE=${VOCAB_SIZE:-5000}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}

# 训练配置
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-6e-4}
MIN_LR=${MIN_LR:-6e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}

# 优化器配置
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# 输出配置
OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_megatron"}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-1000}

# ==================== 打印配置 ====================

echo "================================"
echo "Megatron-LM Training"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "集群配置:"
echo "  节点数: $NUM_NODES"
echo "  当前节点rank: $NODE_RANK"
echo "  GPUs per node: $NUM_GPUS"
echo "  总GPU数: $((NUM_NODES * NUM_GPUS))"
if [ $NUM_NODES -gt 1 ]; then
    echo "  Master地址: $MASTER_ADDR:$MASTER_PORT"
fi
echo ""
echo "模型配置:"
echo "  Vocab size: $VOCAB_SIZE"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Num layers: $NUM_LAYERS"
echo "  Num heads: $NUM_HEADS"
echo "  Sequence length: $MAX_SEQ_LEN"
if [ -f "$CONFIG_FILE" ]; then
    echo "  配置文件: $CONFIG_FILE"
fi
echo ""
echo "训练配置:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Global batch size: $((BATCH_SIZE * NUM_NODES * NUM_GPUS * GRAD_ACCUM))"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LR -> $MIN_LR"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# ==================== 启动训练 ====================
TR=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/torchrun
# 单节点训练
if [ $NUM_NODES -eq 1 ]; then
    $TR --nproc_per_node=$NUM_GPUS \
        train.py \
        --vocab_size $VOCAB_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --num_heads $NUM_HEADS \
        --max_seq_len $MAX_SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --learning_rate $LR \
        --min_lr $MIN_LR \
        --warmup_steps $WARMUP_STEPS \
        --weight_decay $WEIGHT_DECAY \
        --adam_beta1 $ADAM_BETA1 \
        --adam_beta2 $ADAM_BETA2 \
        --max_grad_norm $MAX_GRAD_NORM \
        --fp16 \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --output_dir $OUTPUT_DIR \
        --trainer megatron_trainer
else
    # 多节点训练
    $TR \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --vocab_size $VOCAB_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --num_heads $NUM_HEADS \
        --max_seq_len $MAX_SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --learning_rate $LR \
        --min_lr $MIN_LR \
        --warmup_steps $WARMUP_STEPS \
        --weight_decay $WEIGHT_DECAY \
        --adam_beta1 $ADAM_BETA1 \
        --adam_beta2 $ADAM_BETA2 \
        --max_grad_norm $MAX_GRAD_NORM \
        --fp16 \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --output_dir $OUTPUT_DIR \
        --trainer megatron_trainer
fi

echo ""
echo "================================"
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "================================"
