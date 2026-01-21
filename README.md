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
├── train.py                   # 标准训练脚本
│
├── megatron_model.py          # Megatron模型
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
├── src/                     # 工具
│   ├── monitor.py
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


### 2. 监控训练

```bash
# 实时监控
python tools/monitor.py

# 查看日志
tail -f output_*/train.log

# GPU状态
watch -n 1 nvidia-smi
```

```

###  多节点训练

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
