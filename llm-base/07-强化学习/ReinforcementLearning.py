import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成训练数据
def generate_data(num_samples):
    # 生成随机的二维输入数据，范围在 0 到 10 之间
    state = torch.tensor(np.random.randint(0, 10, (num_samples, 2)), dtype=torch.float32)
    # 计算加法目标值
    target_add = state[:, 0] + state[:, 1]
    # 计算减法目标值
    target_sub = state[:, 0] - state[:, 1]
    return state, target_add, target_sub

# 定义强化学习模型
class ReinforceModel(nn.Module):
    def __init__(self):
        super(ReinforceModel, self).__init__()
        # 输入层到隐藏层的全连接层，输入维度 2，输出维度 5
        self.hidden = nn.Linear(2, 5)
        # 隐藏层到输出层的全连接层，输入维度 5，输出维度 1
        self.output = nn.Linear(5, 1)
        # ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据通过隐藏层并经过 ReLU 激活
        x = self.relu(self.hidden(x))
        # 激活后的数据通过输出层
        x = self.output(x)
        return x

# 定义模型和优化器
model = ReinforceModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义奖励机制
def compute_reward(pred, target):
    # 负的绝对差值作为惩罚，预测越准确奖励越高
    return -torch.abs(pred - target)

# 训练强化学习模型
for epoch in range(1000):
    # 每次生成一个数据样本
    state, target_add, target_sub = generate_data(1)
    # 清空优化器中的梯度信息
    optimizer.zero_grad()

    # 加法学习
    prediction = model(state)
    reward = compute_reward(prediction, target_add)

    # 负的损失用于梯度提升，因为强化学习要最大化奖励
    loss = -reward

    # 反向传播计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

    # 每 100 个 epoch 打印一次奖励信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Reward: {reward.item():.4f}')

# 测试加法效果
x_test = torch.tensor([[5, 3]], dtype=torch.float32)
predicted = model(x_test).item()
print(f"5 + 3 = {predicted:.2f}")
