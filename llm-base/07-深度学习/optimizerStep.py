import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 5)  # 输入层 -> 隐藏层
        self.output = nn.Linear(5, 1)  # 隐藏层 -> 输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))  # 隐藏层
        x = self.output(x)  # 输出层
        return x


# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成训练数据
x_train = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
y_train = torch.tensor([[5.0]], dtype=torch.float32)

# **训练前打印参数**
print("Before Training:")
for name, param in model.named_parameters():
    print(f"{name} - Value:\n{param.data}\n")

# 前向传播
y_pred = model(x_train)
loss = criterion(y_pred, y_train)

# 反向传播
optimizer.zero_grad()
loss.backward()

# **打印梯度**
print("\nGradients (Before Update):")
for name, param in model.named_parameters():
    print(f"{name} - Grad:\n{param.grad}\n")

# 更新参数
optimizer.step()

# **训练后打印参数**
print("\nAfter Training:")
for name, param in model.named_parameters():
    print(f"{name} - Updated Value:\n{param.data}\n")
