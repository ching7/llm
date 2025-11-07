import torch
import time

print("=" * 60)
print("CPU vs GPU (MPS) 计算性能对比")
print("=" * 60)

# 检查可用设备
device_cpu = torch.device("cpu")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_gpu = torch.device("mps")
    gpu_name = "MPS (AMD GPU)"
else:
    device_gpu = None
    gpu_name = "GPU 不可用"

print(f"\nCPU 设备: {device_cpu}")
if device_gpu:
    print(f"GPU 设备: {device_gpu} ({gpu_name})")
else:
    print("GPU 设备: 不可用")
print()

# 定义测试函数
def benchmark_operation(operation_name, operation_func, device, warmup=5, runs=10):
    """性能测试函数"""
    # 预热
    for _ in range(warmup):
        operation_func(device)
    
    # 同步（确保 GPU 操作完成）
    if device.type != "cpu":
        torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
    
    # 正式测试
    start_time = time.time()
    result = None
    for _ in range(runs):
        result = operation_func(device)
        if device.type != "cpu":
            torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / runs
    return avg_time, result

# 测试1: 矩阵乘法
print("测试1: 大规模矩阵乘法 (1000x1000)")
print("-" * 60)

def matrix_multiplication(device):
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    return torch.matmul(a, b)

cpu_time, _ = benchmark_operation("矩阵乘法", matrix_multiplication, device_cpu)
print(f"CPU 平均时间: {cpu_time*1000:.2f} ms")

if device_gpu:
    gpu_time, _ = benchmark_operation("矩阵乘法", matrix_multiplication, device_gpu)
    print(f"GPU 平均时间: {gpu_time*1000:.2f} ms")
    speedup = cpu_time / gpu_time
    print(f"加速比: {speedup:.2f}x")
else:
    print("GPU 不可用，跳过测试")

print()

# 测试2: 元素级运算
print("测试2: 元素级运算 (10000x10000)")
print("-" * 60)

def elementwise_operations(device):
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    return a * b + torch.sin(a) + torch.exp(b / 10)

cpu_time, _ = benchmark_operation("元素级运算", elementwise_operations, device_cpu)
print(f"CPU 平均时间: {cpu_time*1000:.2f} ms")

if device_gpu:
    gpu_time, _ = benchmark_operation("元素级运算", elementwise_operations, device_gpu)
    print(f"GPU 平均时间: {gpu_time*1000:.2f} ms")
    speedup = cpu_time / gpu_time
    print(f"加速比: {speedup:.2f}x")
else:
    print("GPU 不可用，跳过测试")

print()

# 测试3: 卷积操作（模拟神经网络）
print("测试3: 卷积操作 (模拟 CNN)")
print("-" * 60)

def convolution_operation(device):
    # 模拟卷积层: batch_size=32, channels=64, height=224, width=224
    x = torch.randn(32, 64, 224, 224, device=device)
    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
    return conv(x)

cpu_time, _ = benchmark_operation("卷积操作", convolution_operation, device_cpu)
print(f"CPU 平均时间: {cpu_time*1000:.2f} ms")

if device_gpu:
    gpu_time, _ = benchmark_operation("卷积操作", convolution_operation, device_gpu)
    print(f"GPU 平均时间: {gpu_time*1000:.2f} ms")
    speedup = cpu_time / gpu_time
    print(f"加速比: {speedup:.2f}x")
else:
    print("GPU 不可用，跳过测试")

print()

# 测试4: 批量矩阵乘法
print("测试4: 批量矩阵乘法 (batch_size=100, 500x500)")
print("-" * 60)

def batch_matrix_multiplication(device):
    a = torch.randn(100, 500, 500, device=device)
    b = torch.randn(100, 500, 500, device=device)
    return torch.bmm(a, b)

cpu_time, _ = benchmark_operation("批量矩阵乘法", batch_matrix_multiplication, device_cpu)
print(f"CPU 平均时间: {cpu_time*1000:.2f} ms")

if device_gpu:
    gpu_time, _ = benchmark_operation("批量矩阵乘法", batch_matrix_multiplication, device_gpu)
    print(f"GPU 平均时间: {gpu_time*1000:.2f} ms")
    speedup = cpu_time / gpu_time
    print(f"加速比: {speedup:.2f}x")
else:
    print("GPU 不可用，跳过测试")

print()

# 总结
print("=" * 60)
print("性能测试总结")
print("=" * 60)
print("GPU 在并行计算密集型任务上通常比 CPU 快很多倍")
print("特别是矩阵运算、卷积等操作，GPU 的优势更明显")
print("=" * 60)

