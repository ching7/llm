import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的2层神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 输入层 → 隐藏层（2 -> 5）
        self.hidden = nn.Linear(2, 5)
        # 隐藏层 → 输出层（5 -> 1）
        self.output = nn.Linear(5, 1)
        # 激活函数（ReLU）
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据通过隐藏层，进行线性加权求和
        hidden_output = self.hidden(x)
        print(f"Hidden layer output: {hidden_output}")  # 打印隐藏层输出

        # 通过激活函数，进行非线性转换
        activated_output = self.relu(hidden_output)
        print(f"Activated output after ReLU: {activated_output}")  # 打印激活后的输出

        # 最后通过输出层
        final_output = self.output(activated_output)
        print(f"Final output (after output layer): {final_output}")  # 打印输出层结果

        return final_output


# 创建模型实例
model = SimpleNN()

# 训练数据：假设我们有一些输入数据和对应的标签
# 假设我们有 5 个样本，每个样本有 2 个特征
x_train = torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [1.0, 5.0], [3.0, 3.0]], dtype=torch.float32)
y_train = torch.tensor([[3.0], [5.0], [9.0], [6.0], [6.0]], dtype=torch.float32)  # 假设目标是加法操作

# 定义损失函数（均方误差）
criterion = nn.MSELoss()
# 定义优化器（Adam 优化器）
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练过程
for epoch in range(1000):
    model.train()  # 设置模型为训练模式

    # 前向传播：获取预测结果
    output = model(x_train)

    # 计算损失
    loss = criterion(output, y_train)

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每 100 个 epoch 打印一次训练过程的信息
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# 测试阶段
model.eval()  # 设置模型为评估模式
x_test = torch.tensor([[5.0, 3.0]], dtype=torch.float32)  # 测试数据
predicted = model(x_test)  # 获取模型预测结果
print(f"Prediction for input [5.0, 3.0]: {predicted.item():.2f}")
