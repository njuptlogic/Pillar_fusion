#import torch
#torch.cuda.empty_cache()
import torch

# 检查是否有 GPU 可用
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # 打印第一个 GPU 的名称
else:
    print("CUDA is not available. Using CPU.")
import subprocess

def check_gpu_usage():
    try:
        # 执行基本的 nvidia-smi 命令，不使用 --query-gpu 选项
        result = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits"])
        
        # 解码输出为字符串并按行分割
        result = result.decode("utf-8").strip().split("\n")
        
        # 检查每个 GPU 的状态
        for gpu_info in result:
            gpu_data = gpu_info.split(" | ")
            
            # 如果该行数据的字段数不够，跳过
            if len(gpu_data) < 5:
                print("Skipping incomplete line: ", gpu_info)
                continue
            
            gpu_index = gpu_data[0]
            gpu_name = gpu_data[1]
            gpu_util = gpu_data[2].strip(" %")  # 去掉百分号
            mem_free = gpu_data[3].split()[0]  # 获取空闲内存值
            mem_used = gpu_data[4].split()[0]  # 获取已用内存值
            
            # 打印 GPU 使用情况
            print(f"GPU {gpu_index} ({gpu_name}):")
            print(f"  GPU Utilization: {gpu_util}%")
            print(f"  Memory Usage: {mem_used} MB / Total Memory")
            print(f"  Free Memory: {mem_free} MB")
            print("-" * 50)
    
    except subprocess.CalledProcessError as e:
        print("Failed to execute nvidia-smi. Please ensure NVIDIA drivers are installed and configured.")
        print(f"Error: {e}")

# 检查 GPU 使用情况
check_gpu_usage()


