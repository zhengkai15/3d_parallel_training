#!/bin/bash

# 3D并行训练启动脚本 - 调试版本

# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}
PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE:-"gpipe"} #gpipe 1f1b

VOCAB_SIZE=${VOCAB_SIZE:-30000}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}

BATCH_SIZE=${BATCH_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-100}
LR=${LR:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-2}

OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_3d_parallel"}

# ==================== 验证配置 ====================

DP_SIZE=$((NUM_GPUS / (TP_SIZE * PP_SIZE)))

if [ $((DP_SIZE * TP_SIZE * PP_SIZE)) -ne $NUM_GPUS ]; then
    echo "错误: NUM_GPUS ($NUM_GPUS) 必须能被 TP_SIZE ($TP_SIZE) * PP_SIZE ($PP_SIZE) 整除"
    exit 1
fi

# ==================== 打印配置 ====================

echo "================================"
echo "3D Parallel Training (DEBUG MODE)"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "GPU配置:"
echo "  总GPU数: $NUM_GPUS"
echo "  数据并行 (DP): $DP_SIZE"
echo "  张量并行 (TP): $TP_SIZE"
echo "  流水线并行 (PP): $PP_SIZE"
echo ""
echo "Pipeline配置:"
echo "  Microbatches: $NUM_MICROBATCHES"
echo "  调度策略: $PIPELINE_SCHEDULE"
echo ""
echo "模型配置:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Heads: $NUM_HEADS"
echo ""
echo "训练配置:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo ""
echo "================================"
echo ""

# ==================== 启动训练 ====================

TR=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/torchrun

$TR --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_3d_parallel.py \
    --vocab_size $VOCAB_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --max_seq_len $MAX_SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --tp_size $TP_SIZE \
    --pp_size $PP_SIZE \
    --num_microbatches $NUM_MICROBATCHES \
    --pipeline_schedule $PIPELINE_SCHEDULE \
    --logging_steps 10 \
    --output_dir $OUTPUT_DIR

echo ""
echo "================================"
echo "训练完成！"
echo "================================"