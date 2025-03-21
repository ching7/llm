import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义一个简单的2层神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()  # 简化 super 调用，适用于 Python 3
        # 输入层 → 隐藏层（2 -> 5）
        self.hidden = nn.Linear(2, 5)
        # 隐藏层 → 输出层（5 -> 1）
        self.output = nn.Linear(5, 1)
        # 激活函数（ReLU）
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))  # 隐藏层通过 ReLU 激活
        x = self.output(x)  # 输出层直接输出
        return x


# 定义模型
model = SimpleNN()

# 定义损失函数（均方误差）
criterion = nn.MSELoss()

# 定义优化器（随机梯度下降）
# 替换为 Adagrad 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 定义 StepLR 调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


# 生成训练数据（加法和减法）
# 神经元个数
def generate_data(num_samples=1000):
    x = np.random.randint(0, 10, (num_samples, 2))
    y_add = x[:, 0] + x[:, 1]
    y_sub = x[:, 0] - x[:, 1]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y_add, dtype=torch.float32), torch.tensor(y_sub,
                                                                                                        dtype=torch.float32)


# 生成训练集
x_train, y_train_add, y_train_sub = generate_data()

# 训练神经网络（1000个Epoch）
for epoch in range(1000):
    optimizer.zero_grad()  # 清空梯度

    # 前向传播（加法）
    output = model(x_train)

    # 损失函数（学习加法）
    loss = criterion(output.squeeze(), y_train_add)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()
    # 更新学习率
    scheduler.step()

    # 打印权重和偏置参数
    print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
    print("Hidden layer weights:", model.hidden.weight.data)
    print("Hidden layer bias:", model.hidden.bias.data)
    print("Output layer weights:", model.output.weight.data)
    print("Output layer bias:", model.output.bias.data)

# 测试加法效果
x_test = torch.tensor([[5, 3]], dtype=torch.float32)
predicted = model(x_test).item()
print(f"5 + 3 = {predicted:.2f}")
