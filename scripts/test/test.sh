# !/bin/bash
set -x
# 1. 测试P2P通信
TR=/cpfs01/projects-SSD/cfff-4a8d9af84f66_SSD/public/zhengkai/miniconda3/envs/3dparallel/bin/torchrun

$TR  --nproc_per_node=4 test_p2p_comm.py

# 2. 测试简单PP（2GPU）
$TR  --nproc_per_node=2 test_simple_p2p.py