#!/bin/bash

# 3D并行训练启动脚本 - PyTorch DDP版本 (支持多机)
# 使用方法: 
#   单机: bash scripts/run_pytorch_ddp.sh
#   多机手动: 
#     Node 0: NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 bash scripts/run_pytorch_ddp.sh
#     Node 1: NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.1 bash scripts/run_pytorch_ddp.sh

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ==================== 配置参数 ====================

# 1. 硬件与环境配置 (支持环境变量覆盖)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)} # 自动检测GPU数量
NNODES=${NNODES:-1}                            # 总机器数，默认为1
NODE_RANK=${NODE_RANK:-0}                      # 当前机器排名，默认为0
MASTER_ADDR=${MASTER_ADDR:-"localhost"}        # 主节点IP
MASTER_PORT=${MASTER_PORT:-29500}              # 通信端口

# 2. 模型配置
TP_SIZE=${TP_SIZE:-2}
VOCAB_SIZE=${VOCAB_SIZE:-30000}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}

# 3. 训练配置
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_STEPS=${MAX_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# 4. 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"./aexp/output_pytorch_ddp"}

# ==================== 打印信息 ====================

echo "================================"
echo "3D Parallel Training with PyTorch DDP (Multi-Node Ready)"
echo "================================"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "分布式环境配置:"
echo "  Nodes (NNODES): $NNODES"
echo "  Current Rank (NODE_RANK): $NODE_RANK"
echo "  Master Addr: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  GPUs per Node: $NUM_GPUS"
echo ""
echo "模型配置:"
echo "  TP Size: $TP_SIZE"
echo "  Hidden size: $HIDDEN_SIZE"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# ==================== 启动命令 ====================

# 定义Python解释器路径 (根据实际情况修改)
PY="/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/torchrun"

# 构建 torchrun 命令
# 核心改变: 增加了 --nnodes, --node_rank, --master_addr, --master_port
CMD="$PY \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NNODES \
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
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --tp_size $TP_SIZE \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir $OUTPUT_DIR"

# 打印完整命令以便调试
echo "Executing command:"
echo "$CMD"
echo ""

# 执行命令
$CMD

echo ""
echo "================================"
echo "节点 $NODE_RANK 任务结束"
echo "================================"
