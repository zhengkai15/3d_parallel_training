#!/bin/bash

# 单GPU训练脚本 - 用于快速测试
# 使用方法: bash scripts/run_single_gpu.sh

set -x

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

# 小模型配置 (快速测试)
VOCAB_SIZE=${VOCAB_SIZE:-5000}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}
NUM_SAMPLES=${NUM_SAMPLES:-1000}

# 训练配置
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_STEPS=${MAX_STEPS:-1000}
LR=${LR:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-10}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_single_gpu"}

# ==================== 打印配置 ====================

echo "================================"
echo "Single GPU Training (Test Mode)"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "模型配置 (小模型测试):"
echo "  Vocab size: $VOCAB_SIZE"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Heads: $NUM_HEADS"
echo "  Sequence length: $MAX_SEQ_LEN"
echo ""
echo "训练配置:"
echo "  Samples: $NUM_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# ==================== 启动训练 ====================

PY=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/python
$PY train.py \
    --vocab_size $VOCAB_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --max_seq_len $MAX_SEQ_LEN \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 5 \
    --save_steps 50 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --trainer simple

echo ""
echo "================================"
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "================================"
