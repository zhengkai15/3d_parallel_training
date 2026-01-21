#!/bin/bash

# 3D并行训练启动脚本 - DeepSpeed版本
# 使用方法: bash scripts/run_deepspeed.sh

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

# GPU配置
NUM_GPUS=${NUM_GPUS:-4}

# 并行配置
TP_SIZE=${TP_SIZE:-2}
ZERO_STAGE=${ZERO_STAGE:-2}  # ZeRO优化阶段 (0, 1, 2, 3)

# 模型配置
VOCAB_SIZE=${VOCAB_SIZE:-5000}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}

# 训练配置
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_STEPS=${MAX_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_deepspeed_zero${ZERO_STAGE}"}

# DeepSpeed配置文件
if [ $ZERO_STAGE -eq 3 ]; then
    DS_CONFIG="configs/deepspeed/ds_config_zero3.json"
else
    DS_CONFIG="configs/deepspeed/ds_config_zero2.json"
fi

# 检查配置文件是否存在
if [ ! -f "$DS_CONFIG" ]; then
    echo "错误: DeepSpeed配置文件不存在: $DS_CONFIG"
    echo "请确保配置文件在正确位置"
    exit 1
fi

# ==================== 打印配置 ====================

echo "================================"
echo "3D Parallel Training with DeepSpeed"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "GPU配置:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  ZeRO Stage: $ZERO_STAGE"
echo ""
echo "模型配置:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo ""
echo "配置文件: $DS_CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# ==================== 启动训练 ====================
DP=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/deepspeed
$DP --num_gpus=$NUM_GPUS \
    train.py \
    --deepspeed_config $DS_CONFIG \
    --vocab_size $VOCAB_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --max_seq_len $MAX_SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --tp_size $TP_SIZE \
    --zero_stage $ZERO_STAGE \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir $OUTPUT_DIR \
    --trainer deepspeed

echo ""
echo "================================"
echo "Training completed!"
echo "模型保存在: $OUTPUT_DIR"
echo "================================"
