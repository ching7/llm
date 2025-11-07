import torch
import subprocess
import re

# check pytorch version
print(f"PyTorch version: {torch.__version__}")

# 方法1: 创建时直接指定设备（推荐）
# 自动选择最佳可用设备
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"使用设备: {device}")

# 创建时直接指定设备
x = torch.tensor([[2, 3]], device=device)
print("成功创建了一个张量（直接在指定设备上）:")
print(x)

# 判断 tensor 是在 GPU 还是 CPU 上
print(f"\n张量设备信息:")
print(f"设备位置: {x.device}")
print(f"是否在 CPU: {x.device.type == 'cpu'}")
print(f"是否在 CUDA (NVIDIA GPU): {x.is_cuda}")
if hasattr(x, "is_mps"):
    print(f"是否在 MPS (macOS/AMD GPU): {x.is_mps}")

# 根据设备类型显示详细信息
if x.device.type == "cpu":
    print("✓ 张量当前在 CPU 上")
elif x.is_cuda:
    print(f"✓ 张量当前在 CUDA GPU 上 (设备 {x.device.index})")
elif hasattr(x, "is_mps") and x.is_mps:
    print("✓ 张量当前在 MPS (Metal/AMD GPU) 上")
else:
    print(f"✓ 张量当前在 {x.device.type} 设备上")

# 方法2: 使用字符串直接指定设备（更简洁）
print(f"\n方法2: 使用字符串直接指定设备")
x_mps = torch.tensor(
    [[4, 5]],
    device=(
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    ),
)
print(f"直接在 MPS 上创建的张量: {x_mps}")
print(f"设备: {x_mps.device}")

# 方法3: 使用 torch 的默认设备设置（全局设置）
print(f"\n方法3: 设置默认设备")
torch.set_default_device(device)
x_default = torch.tensor([[6, 7]])
print(f"使用默认设备创建的张量: {x_default}")
print(f"设备: {x_default.device}")
