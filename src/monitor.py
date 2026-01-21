"""
训练监控工具
实时显示GPU使用情况和训练进度
"""
import subprocess
import time
import os
from datetime import datetime


def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n')
    except:
        return []


def format_memory(mb):
    """格式化内存显示"""
    if mb >= 1024:
        return f"{mb / 1024:.1f}GB"
    return f"{mb:.0f}MB"


def display_gpu_status():
    """显示GPU状态"""
    gpu_info = get_gpu_info()

    print("\n" + "=" * 100)
    print(f"{'GPU':<5} {'Name':<25} {'Util':<10} {'Memory':<20} {'Temp':<10}")
    print("=" * 100)

    for line in gpu_info:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            gpu_id, name, util, mem_used, mem_total, temp = parts
            mem_used = float(mem_used)
            mem_total = float(mem_total)
            mem_percent = (mem_used / mem_total) * 100

            print(f"{gpu_id:<5} {name:<25} {util:>3}%      "
                  f"{format_memory(mem_used):>7}/{format_memory(mem_total):<7} ({mem_percent:>5.1f}%)  "
                  f"{temp:>3}°C")

    print("=" * 100)


def check_training_processes():
    """检查正在运行的训练进程"""
    try:
        # 查找Python训练进程
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        processes = []
        for line in result.stdout.split('\n'):
            if 'train.py' in line or 'deepspeed' in line or 'torchrun' in line:
                processes.append(line)

        if processes:
            print("\nRunning Training Processes:")
            print("-" * 100)
            for proc in processes:
                # 简化显示
                parts = proc.split()
                if len(parts) > 10:
                    print(f"PID: {parts[1]}, CPU: {parts[2]}%, MEM: {parts[3]}%, CMD: {' '.join(parts[10:13])}")
        else:
            print("\nNo training processes found.")

    except:
        pass


def check_output_logs(output_dir='./output_deepspeed_zero2'):
    """检查输出日志"""
    if not os.path.exists(output_dir):
        print(f"\nOutput directory not found: {output_dir}")
        return

    print(f"\nChecking logs in: {output_dir}")

    # 查找最新的checkpoint
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoints.append(item)

    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        print(f"Latest checkpoint: {checkpoints[-1]}")
    else:
        print("No checkpoints found yet.")

    # 显示训练日志最后几行
    log_files = ['train.log', 'deepspeed_log.txt']
    for log_file in log_files:
        log_path = os.path.join(output_dir, log_file)
        if os.path.exists(log_path):
            print(f"\nLast 5 lines of {log_file}:")
            print("-" * 100)
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(line.strip())
            except:
                pass


def monitor_loop(interval=5, output_dir='./output_deepspeed_zero2'):
    """持续监控循环"""
    print("\n" + "=" * 100)
    print("3D Parallel Training Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 100)

    try:
        while True:
            # 清屏 (可选)
            # os.system('clear')

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

            # 显示GPU状态
            display_gpu_status()

            # 检查训练进程
            check_training_processes()

            # 检查输出日志
            check_output_logs(output_dir)

            # 等待
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Monitor")
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--output_dir', type=str, default='./output_deepspeed_zero2',
                        help='Output directory to monitor')
    parser.add_argument('--once', action='store_true', help='Run once and exit')

    args = parser.parse_args()

    if args.once:
        # 只运行一次
        display_gpu_status()
        check_training_processes()
        check_output_logs(args.output_dir)
    else:
        # 持续监控
        monitor_loop(args.interval, args.output_dir)
