# 3D并行训练完整使用指南

本指南详细介绍如何使用本项目进行大规模语言模型训练。

## 目录

1. [快速开始](#快速开始)
2. [训练模式](#训练模式)
3. [并行策略选择](#并行策略选择)
4. [多节点训练](#多节点训练)
5. [性能优化](#性能优化)
6. [常见问题](#常见问题)

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 测试环境
bash quick_test.sh
```

### 2. 单GPU测试

```bash
# 最小配置测试
python train.py \
    --vocab_size 5000 \
    --hidden_size 256 \
    --num_layers 4 \
    --batch_size 4 \
    --max_steps 100
```

### 3. 多GPU训练

```bash
# 4 GPU DDP训练
bash run_pytorch_ddp.sh
```

---

## 训练模式

### 模式1: 标准训练 (train.py)

**适用场景**:
- 小模型 (< 1B参数)
- 简单的数据并行或张量并行

**启动方式**:
```bash
# DDP (数据并行)
torchrun --nproc_per_node=4 train.py --batch_size 8

# DDP + TP (数据+张量并行)
torchrun --nproc_per_node=4 train.py --tp_size 2 --batch_size 8
```

### 模式2: 3D并行训练 (train_3d_parallel.py)

**适用场景**:
- 中大型模型 (1B - 100B参数)
- 需要流水线并行
- 多维度并行组合

**启动方式**:
```bash
# 使用启动脚本（推荐）
bash run_3d_parallel.sh

# 或直接使用torchrun
torchrun --nproc_per_node=8 train_3d_parallel.py \
    --tp_size 2 \
    --pp_size 2 \
    --num_microbatches 4 \
    --pipeline_schedule 1f1b
```

**3D并行配置示例**:

| GPU数 | DP | TP | PP | 适用模型大小 |
|-------|----|----|----| ------------|
| 4     | 4  | 1  | 1  | < 1B        |
| 4     | 2  | 2  | 1  | 1-3B        |
| 8     | 2  | 2  | 2  | 3-7B        |
| 16    | 2  | 4  | 2  | 7-13B       |
| 32    | 2  | 4  | 4  | 13-30B      |
| 64    | 2  | 8  | 4  | 30-70B      |

### 模式3: Megatron训练 (train_megatron.py)

**适用场景**:
- 使用Megatron-LM的高级优化
- 需要更精细的张量并行控制
- 追求极致性能

**启动方式**:
```bash
# 使用启动脚本
bash run_megatron.sh

# 使用配置文件
CONFIG_FILE=configs/medium_model.yaml bash run_megatron.sh

# 自定义参数
NUM_GPUS=8 \
HIDDEN_SIZE=1024 \
NUM_LAYERS=24 \
bash run_megatron.sh
```

---

## 并行策略选择

### 决策树

```
开始
 |
 ├─ 模型能放入单GPU？
 |   ├─ 是 → 使用 DDP (纯数据并行)
 |   └─ 否 → 继续
 |
 ├─ 单层能放入单GPU？
 |   ├─ 否 → 必须使用 TP (张量并行)
 |   └─ 是 → 继续
 |
 ├─ 模型非常深 (> 30层)？
 |   ├─ 是 → 考虑 PP (流水线并行)
 |   └─ 否 → 继续
 |
 └─ 选择组合:
     ├─ DP only: 最简单，通信开销最小
     ├─ DP + TP: 适合中等模型
     ├─ DP + PP: 适合非常深的模型
     └─ DP + TP + PP: 适合超大模型
```

### 并行方式对比

| 并行方式 | 优点 | 缺点 | 通信开销 | 内存节省 |
|---------|------|------|----------|---------|
| **DP** | 简单，高效 | 每GPU需完整模型 | 低 | 无 |
| **TP** | 可训练超大层 | 通信频繁 | 高 | 高 |
| **PP** | 内存效率高 | Bubble开销 | 中 | 高 |
| **ZeRO-2** | 节省内存 | 通信开销 | 中 | 中 |
| **ZeRO-3** | 最省内存 | 通信开销大 | 高 | 很高 |

### 配置建议

#### 小模型 (< 1B参数, 如GPT-2)
```bash
NUM_GPUS=4
TP_SIZE=1
PP_SIZE=1
# DP_SIZE = 4
```

#### 中型模型 (1-10B参数, 如GPT-2 Large)
```bash
NUM_GPUS=8
TP_SIZE=2
PP_SIZE=2
# DP_SIZE = 2
```

#### 大型模型 (10-100B参数, 如GPT-3)
```bash
NUM_GPUS=64
TP_SIZE=8
PP_SIZE=4
# DP_SIZE = 2
```

#### 超大模型 (> 100B参数, 如GPT-3 175B)
```bash
NUM_GPUS=128
TP_SIZE=8
PP_SIZE=8
# DP_SIZE = 2

# 配合ZeRO-3和CPU offload
```

---

## 多节点训练

### 单节点多GPU

```bash
# 在单个节点上使用所有GPU
NUM_GPUS=8 bash run_3d_parallel.sh
```

### 多节点训练

#### 方案1: 使用torchrun

**节点0 (主节点)**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train_3d_parallel.py \
    --tp_size 2 \
    --pp_size 2
```

**节点1**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train_3d_parallel.py \
    --tp_size 2 \
    --pp_size 2
```

#### 方案2: 使用多节点脚本

**节点0**:
```bash
NUM_NODES=2 \
NODE_RANK=0 \
MASTER_ADDR="192.168.1.100" \
bash run_multinode.sh
```

**节点1**:
```bash
NUM_NODES=2 \
NODE_RANK=1 \
MASTER_ADDR="192.168.1.100" \
bash run_multinode.sh
```

#### 方案3: 使用SLURM

```bash
#!/bin/bash
#SBATCH --job-name=3d_parallel
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

# 设置主节点
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# 启动训练
srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_3d_parallel.py \
    --tp_size 2 \
    --pp_size 2
```

---

## 性能优化

### 1. 通信优化

#### 启用NCCL优化
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 启用InfiniBand
export NCCL_NET_GDR_LEVEL=5       # 启用GPUDirect RDMA
export NCCL_SOCKET_IFNAME=eth0    # 指定网络接口
```

#### 减少通信开销
- 增加梯度累积步数
- 使用更大的batch size
- 减少TP size（如果可能）

### 2. 内存优化

#### 启用激活重计算
```python
# 在模型中启用gradient checkpointing
model.gradient_checkpointing_enable()
```

#### 使用ZeRO优化
```bash
# DeepSpeed配置
bash run_deepspeed.sh
```

#### 调整batch size和序列长度
```bash
# 减小batch size
--batch_size 4

# 增加梯度累积
--gradient_accumulation_steps 8

# 减小序列长度（如果可以）
--max_seq_len 1024
```

### 3. 计算优化

#### 使用混合精度
```bash
# FP16 (推荐用于Volta/Turing架构)
--fp16

# BF16 (推荐用于Ampere及以上架构)
--bf16
```

#### 启用编译优化
```python
# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)
```

#### 使用FlashAttention
```python
# 安装 flash-attn
pip install flash-attn

# 在模型中使用
from flash_attn import flash_attn_func
```

### 4. 数据加载优化

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,      # 增加worker数量
    pin_memory=True,    # 启用pin memory
    prefetch_factor=2   # 预取下一个batch
)
```

### 5. Pipeline并行优化

#### 选择合适的调度策略
```bash
# 1F1B (推荐) - 更低的内存占用，更高的效率
--pipeline_schedule 1f1b

# GPipe - 更简单，但bubble开销更大
--pipeline_schedule gpipe
```

#### 调整microbatch数量
```bash
# 增加microbatch可以减少bubble，但增加内存
--num_microbatches 8
```

---

## 性能基准测试

### 运行benchmark

```bash
# 测试所有配置
NUM_GPUS=8 bash benchmark.sh

# 查看结果
cat benchmark_results/summary.csv

# 生成可视化
cd benchmark_results && python plot_results.py
```

### 性能指标

监控以下指标:
- **Samples/sec**: 每秒处理样本数
- **Tokens/sec**: 每秒处理token数
- **GPU利用率**: 目标 > 80%
- **内存使用**: 避免OOM
- **通信时间**: 应 < 20%总时间

---

## 常见问题

### 1. OOM (Out of Memory)

**症状**: CUDA out of memory错误

**解决方案**:
```bash
# 方案1: 减小batch size
--batch_size 2

# 方案2: 增加梯度累积
--gradient_accumulation_steps 16

# 方案3: 启用ZeRO-3
bash run_deepspeed.sh

# 方案4: 启用activation checkpointing
model.gradient_checkpointing_enable()

# 方案5: 增加TP或PP size
--tp_size 4 --pp_size 2
```

### 2. 训练速度慢

**症状**: samples/sec很低

**解决方案**:
```bash
# 方案1: 增大batch size
--batch_size 16

# 方案2: 减少TP size（如果内存允许）
--tp_size 1

# 方案3: 使用FP16/BF16
--fp16

# 方案4: 优化数据加载
--num_workers 4

# 方案5: 启用NCCL优化
export NCCL_IB_DISABLE=0
```

### 3. 损失不收敛

**症状**: loss震荡或不下降

**解决方案**:
```bash
# 方案1: 降低学习率
--learning_rate 1e-5

# 方案2: 增加warmup
--warmup_steps 1000

# 方案3: 使用BF16代替FP16
--bf16

# 方案4: 调整梯度裁剪
--max_grad_norm 0.5

# 方案5: 检查数据质量
```

### 4. 通信hang或超时

**症状**: 训练卡住，NCCL timeout

**解决方案**:
```bash
# 方案1: 增加超时时间
export NCCL_TIMEOUT=3600

# 方案2: 检查网络
export NCCL_DEBUG=INFO

# 方案3: 禁用不稳定的网络
export NCCL_IB_DISABLE=1

# 方案4: 使用正确的网络接口
export NCCL_SOCKET_IFNAME=eth0
```

### 5. 多节点训练失败

**症状**: 节点之间无法通信

**检查清单**:
1. 确保所有节点能互相ping通
2. 确保防火墙允许相关端口
3. 确保MASTER_ADDR和MASTER_PORT正确
4. 确保所有节点使用相同的代码和配置
5. 检查NCCL版本一致性

---

## 监控和调试

### 实时监控

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控训练进度
python monitor.py --interval 5

# 查看日志
tail -f output_3d_parallel/train.log
```

### 性能分析

```bash
# PyTorch Profiler
python -m torch.utils.bottleneck train.py

# NVIDIA Nsight Systems
nsys profile -o profile.qdrep \
    python train.py

# 查看profile结果
nsys-ui profile.qdrep
```

---

## 进阶技巧

### 1. 动态loss scaling
```python
scaler = torch.cuda.amp.GradScaler(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)
```

### 2. 学习率调度
```python
# Cosine decay with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### 3. 检查点管理
```bash
# 定期保存
--save_steps 1000

# 只保留最近N个检查点
# 在train.py中实现检查点清理逻辑
```

---

## 总结

### 推荐配置

**开发/测试**:
```bash
bash quick_test.sh
```

**小规模训练 (< 10B参数)**:
```bash
bash run_pytorch_ddp.sh
```

**中等规模训练 (10-100B参数)**:
```bash
bash run_3d_parallel.sh
```

**大规模训练 (> 100B参数)**:
```bash
bash run_megatron.sh
# 或使用DeepSpeed
bash run_deepspeed.sh
```

### 最佳实践

1. **始终从小模型开始测试**
2. **逐步增加并行维度**
3. **监控GPU利用率和内存使用**
4. **定期保存检查点**
5. **记录训练日志和配置**
6. **使用版本控制管理代码**

---

## 参考资源

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

---

**祝训练顺利！** 🚀
