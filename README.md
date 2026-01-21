# 3D并行训练框架

<div align="center">

**工业级大语言模型3D并行训练系统**

支持 Data Parallel (DP) + Tensor Parallel (TP) + Pipeline Parallel (PP)

[快速开始](#-快速开始) • [文档](docs/USAGE_GUIDE.md) • [示例](examples/)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## ✨ 核心特性

| 特性 | 说明 |
|-----|------|
| 🎯 **完整3D并行** | DP + TP (Megatron) + PP (GPipe/1F1B) |
| ⚡ **高性能** | 接近理论加速比，支持混合精度 |
| 🔧 **易用性** | 统一入口脚本，一键启动 |
| 📦 **开箱即用** | 预配置模型，自动优化 |
| 🌐 **多节点** | 支持单机多卡和多机多卡 |
| 🔍 **可观测** | 完整日志、监控、检查点 |

## 📁 项目结构

```
3d_parallel_training/
├── train                      # 🎯 统一入口脚本 (推荐)
├── train.py                   # 标准训练脚本
├── train_3d_parallel.py       # 3D并行训练
├── train_megatron.py          # Megatron训练
│
├── model.py                   # 基础模型
├── megatron_model.py          # Megatron模型
├── pipeline_parallel.py       # Pipeline引擎
│
├── scripts/                   # 启动脚本
│   ├── run_3d_parallel.sh
│   ├── run_megatron.sh
│   ├── run_deepspeed.sh
│   └── run_multinode.sh
│
├── configs/                   # 配置文件
│   ├── models/                # 模型配置
│   │   ├── small.yaml
│   │   ├── medium.yaml
│   │   └── large.yaml
│   └── deepspeed/             # DeepSpeed配置
│       ├── zero2.json
│       └── zero3.json
│
├── tools/                     # 工具
│   ├── monitor.py
│   ├── test_model.py
│   ├── benchmark.sh
│   └── quick_test.sh
│
├── docs/                      # 文档
│   └── USAGE_GUIDE.md
│
└── examples/                  # 示例 (TODO)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 快速测试
bash tools/quick_test.sh
```

### 2. 选择训练模式

#### 方式A: 使用统一入口 (推荐 ⭐)

```bash
# 简单DDP训练 (入门)
python train --mode simple --gpus 4

# 3D并行训练 (推荐)
python train --mode 3d --gpus 8 --tp 2 --pp 2

# Megatron训练 (高性能)
python train --mode megatron --gpus 8

# DeepSpeed训练 (省内存)
python train --mode deepspeed --gpus 4 --zero-stage 2
```

#### 方式B: 使用Shell脚本

```bash
# 3D并行
NUM_GPUS=8 TP_SIZE=2 PP_SIZE=2 bash scripts/run_3d_parallel.sh

# Megatron
NUM_GPUS=8 bash scripts/run_megatron.sh

# DeepSpeed
ZERO_STAGE=2 bash scripts/run_deepspeed.sh
```

#### 方式C: 直接调用Python脚本

```bash
# 3D并行
torchrun --nproc_per_node=8 train_3d_parallel.py \
    --tp_size 2 --pp_size 2 --hidden_size 768

# Megatron
torchrun --nproc_per_node=8 train_megatron.py \
    --hidden_size 1024 --num_layers 24
```

### 3. 监控训练

```bash
# 实时监控
python tools/monitor.py

# 查看日志
tail -f output_*/train.log

# GPU状态
watch -n 1 nvidia-smi
```

## 📊 性能对比

| 配置 | GPU数 | 加速比 | 适用模型 |
|-----|-------|--------|---------|
| DDP | 4 | 3.5x | < 1B |
| DP+TP | 4 | 3.2x | 1-3B |
| DP+TP+PP | 8 | 6.5x | 3-10B |
| 3D并行 | 16 | 12x | 10-30B |
| 3D+ZeRO3 | 32 | 22x | 30-100B |

```bash
# 运行性能测试
NUM_GPUS=8 bash tools/benchmark.sh
```

## 💡 使用示例

### 示例1: 小模型快速训练

```bash
# GPT-2 Small (117M参数)
python train --mode simple --gpus 4 \
    --hidden-size 768 --num-layers 12 \
    --batch-size 8 --max-steps 1000
```

### 示例2: 中型模型训练

```bash
# GPT-2 Medium (345M参数)
python train --mode 3d --gpus 8 --tp 2 --pp 2 \
    --hidden-size 1024 --num-layers 24 \
    --batch-size 4 --max-steps 10000
```

### 示例3: 大模型训练

```bash
# GPT-3 1.3B
python train --mode megatron --gpus 16 \
    --config configs/models/large.yaml
```

### 示例4: 多节点训练

**节点0 (主节点)**:
```bash
NUM_NODES=2 NODE_RANK=0 MASTER_ADDR="192.168.1.100" \
bash scripts/run_multinode.sh
```

**节点1**:
```bash
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR="192.168.1.100" \
bash scripts/run_multinode.sh
```

## ⚙️ 配置指南

### 并行策略选择

```python
# 决策树
if 模型 < 1B:
    使用 DDP (--mode simple)
elif 模型 < 10B:
    使用 DP+TP (--mode 3d --tp 2)
elif 模型 < 100B:
    使用 DP+TP+PP (--mode 3d --tp 4 --pp 2)
else:
    使用 3D+ZeRO3 (--mode deepspeed --zero-stage 3)
```

### 参数建议

| 参数 | 小模型 | 中模型 | 大模型 |
|-----|--------|--------|--------|
| `--tp` | 1 | 2 | 4-8 |
| `--pp` | 1 | 2 | 2-4 |
| `--batch-size` | 8-16 | 4-8 | 2-4 |
| `--zero-stage` | 0 | 2 | 3 |

## 📖 文档

- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - 完整使用指南
  - 详细的并行策略说明
  - 多节点训练配置
  - 性能优化技巧
  - 常见问题解答

## 🛠️ 高级功能

### 混合精度训练

```bash
python train --mode 3d --gpus 8 --fp16
```

### 自定义配置文件

```bash
python train --mode megatron --config my_config.yaml
```

### 从检查点恢复

```bash
python train --mode 3d --resume-from output/checkpoint-1000
```

## 🐛 故障排查

### OOM (内存不足)

```bash
# 方案1: 减小batch size
python train --mode 3d --batch-size 2

# 方案2: 使用ZeRO-3
python train --mode deepspeed --zero-stage 3

# 方案3: 增加并行度
python train --mode 3d --tp 4 --pp 2
```

### 训练速度慢

```bash
# 方案1: 检查GPU利用率
nvidia-smi dmon

# 方案2: 增大batch size
python train --batch-size 16

# 方案3: 使用混合精度
python train --fp16
```

### 通信超时

```bash
# 增加超时时间
export NCCL_TIMEOUT=3600

# 启用调试
export NCCL_DEBUG=INFO
```

## 📞 获取帮助

```bash
# 查看帮助
python train --help

# 查看详细文档
cat docs/USAGE_GUIDE.md

# 运行测试
bash tools/quick_test.sh
```

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

本项目参考了：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [PyTorch](https://pytorch.org/)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！⭐**

Made with ❤️ for the AI community

</div>
