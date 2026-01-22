#!/bin/bash

# 多节点训练启动脚本
# 需要在每个节点上分别执行

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

# 集群配置
NUM_NODES=${NUM_NODES:-2}           # 节点总数
NODE_RANK=${NODE_RANK:-$1}           # 当前节点rank (0, 1, 2, ...)
MASTER_ADDR=${MASTER_ADDR:-"dlc1443fgp02dkod-master-0"}  # 主节点IP
MASTER_PORT=${MASTER_PORT:-56747}    # 主节点端口

# GPU配置
NUM_GPUS=${NUM_GPUS:-4}             # 每个节点的GPU数
NUM_MICROBATCHES=${NUM_MICROBATCHES:-2}
PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE:-"gpipe"} #gpipe 1f1b

# 并行配置
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}

# 模型配置
VOCAB_SIZE=${VOCAB_SIZE:-10000}
HIDDEN_SIZE=${HIDDEN_SIZE:-1536}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}

# 训练配置
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-100000}
LR=${LR:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-2}

OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_multinode"}

# ==================== 验证配置 ====================

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
DP_SIZE=$((TOTAL_GPUS / (TP_SIZE * PP_SIZE)))

echo "================================"
echo "多节点3D并行训练"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "集群配置:"
echo "  节点数: $NUM_NODES"
echo "  当前节点: $NODE_RANK"
echo "  每节点GPU: $NUM_GPUS"
echo "  总GPU数: $TOTAL_GPUS"
echo "  主节点: $MASTER_ADDR:$MASTER_PORT"
echo ""
echo "并行配置:"
echo "  DP size: $DP_SIZE"
echo "  TP size: $TP_SIZE"
echo "  PP size: $PP_SIZE"
echo ""
echo "模型配置:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Heads: $NUM_HEADS"
echo ""
echo "Pipeline配置:"
echo "  Microbatches: $NUM_MICROBATCHES"
echo "  调度策略: $PIPELINE_SCHEDULE"
echo ""
echo "================================"

# ==================== 启动训练 ====================
TR=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/torchrun
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
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --tp_size $TP_SIZE \
    --pp_size $PP_SIZE \
    --num_microbatches $NUM_MICROBATCHES \
    --pipeline_schedule $PIPELINE_SCHEDULE \
    --logging_steps 10 \
    --output_dir $OUTPUT_DIR \
    --trainer 3dtrainer

echo "================================"
echo "节点 $NODE_RANK 训练完成！"
echo "================================"

# bash run_multinode.sh 0
# bash run_multinode.sh 1
