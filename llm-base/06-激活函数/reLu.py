import torch
import torch.nn as nn

# 定义 ReLU
relu = nn.ReLU()
# 输入：5 和 3
input_data = torch.tensor([[5.0, 3.0]])
# 权重初始化
# 修改 W 的形状为 (2, 5)
W = torch.tensor([[1.0, 0.3, -0.2, 0.5, -0.7], [0.5, 0.7, 0.8, -0.5, 0.9]])
b = torch.tensor([0.1, 0.2, -0.1, 0.3, 0.5])
# 线性加权求和
hidden_output = torch.matmul(input_data, W) + b
print(f"加权求和输出: {hidden_output}")
# 通过 ReLU 进行激活
activated_output = relu(hidden_output)
print(f"ReLU 输出: {activated_output}")
