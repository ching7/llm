import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义一个简单的2层神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 5)  # 输入层到隐藏层 (2 -> 5)
        self.output = nn.Linear(5, 1)  # 隐藏层到输出层 (5 -> 1)
        self.relu = nn.ReLU()  # ReLU 激活函数

    def forward(self, x):
        z1 = self.hidden(x)  # 计算隐藏层加权和
        a1 = self.relu(z1)  # 通过 ReLU 激活
        z2 = self.output(a1)  # 计算输出层加权和
        return z2, z1, a1  # 返回所有中间计算结果


# 定义模型
model = SimpleNN()

# 定义损失函数（均方误差）
criterion = nn.MSELoss()

# 定义优化器（Adam）
optimizer = optim.Adam(model.parameters(), lr=0.1)


# 生成训练数据（加法）
def generate_data(num_samples=1000):
    x = np.random.randint(0, 10, (num_samples, 2))
    y_add = x[:, 0] + x[:, 1]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y_add, dtype=torch.float32)


x_train, y_train_add = generate_data()

# 训练神经网络
for epoch in range(1000):
    optimizer.zero_grad()  # 清空梯度

    # 前向传播
    output, z1, a1 = model(x_train)
    loss = criterion(output.squeeze(), y_train_add)  # 计算损失

    # 打印前向传播计算过程
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
        # 隐藏层权重和偏置公式
        hidden_weights = model.hidden.weight.data
        hidden_bias = model.hidden.bias.data
        hidden_formulas = []
        for i in range(5):
            formula = f"z1_{i + 1} = {hidden_weights[i][0].item():.4f} * x1 + {hidden_weights[i][1].item():.4f} * x2 + {hidden_bias[i].item():.4f}"
            hidden_formulas.append(formula)
        print("Hidden layer formulas:")
        for formula in hidden_formulas:
            print(formula)

        # 输出层权重和偏置公式
        output_weights = model.output.weight.data
        output_bias = model.output.bias.data
        output_formula = f"z2 = {output_weights[0][0].item():.4f} * a1_1 + {output_weights[0][1].item():.4f} * a1_2 + {output_weights[0][2].item():.4f} * a1_3 + {output_weights[0][3].item():.4f} * a1_4 + {output_weights[0][4].item():.4f} * a1_5 + {output_bias[0].item():.4f}"
        print("Output layer formula:")
        print(output_formula)

    # 反向传播
    loss.backward()

    # 打印梯度信息
    if epoch % 100 == 0:
        print("Hidden layer weights gradient:", model.hidden.weight.grad)
        print("Hidden layer bias gradient:", model.hidden.bias.grad)
        print("Output layer weights gradient:", model.output.weight.grad)
        print("Output layer bias gradient:", model.output.bias.grad)

    optimizer.step()  # 更新参数

# 测试加法效果
x_test = torch.tensor([[5, 3]], dtype=torch.float32)
predicted, _, _ = model(x_test)
print(f"5 + 3 = {predicted.item():.2f}")

# 在这个神经网络的上下文中，z1、x1、x2、z2 和 a1_1 等符号代表了神经网络中不同层的变量和计算结果，下面为你详细解释：

# 输入层
# x1 和 x2：这两个变量代表输入层的神经元。在这个例子中，输入层有 2 个神经元，
# 因为在 SimpleNN 类的 __init__ 方法里，self.hidden = nn.Linear(2, 5) 表示输入层的维度是 2。
# 训练数据是通过 generate_data 函数生成的，每次输入是一个包含两个元素的向量，这两个元素分别对应 x1 和 x2。例如，当输入 [5, 3] 时，x1 就是 5，x2 就是 3。

# 隐藏层
# z1：z1 是隐藏层神经元的加权和，它是输入层的输出经过隐藏层权重矩阵加权并加上偏置后的结果。
# 在代码中，z1 = self.hidden(x) 这一行实现了这个计算。
# 由于隐藏层有 5 个神经元，所以 z1 实际上是一个包含 5 个元素的向量，分别表示为 z1_1、z1_2、z1_3、z1_4 和 z1_5。
# 每个 z1_i 都由输入层的 x1 和 x2 以及对应的权重和偏置计算得到，例如 z1_1 = -0.6969 * x1 + -0.2592 * x2 + -0.4995。
# a1：a1 是隐藏层神经元经过激活函数（这里是 ReLU 激活函数）处理后的输出。在代码中，a1 = self.relu(z1) 实现了这个操作。
# 同样，a1 也是一个包含 5 个元素的向量，分别表示为 a1_1、a1_2、a1_3、a1_4 和 a1_5。

# 输出层
# z2：z2 是输出层神经元的加权和，它是隐藏层的输出 a1 经过输出层权重矩阵加权并加上偏置后的结果。
# 在代码中，z2 = self.output(a1) 实现了这个计算。这里输出层只有 1 个神经元，所以 z2 是一个标量值。
# 总结来说，这些符号代表了神经网络在进行前向传播过程中不同层的输入、加权和以及激活后的输出，通过这些计算，神经网络可以从输入数据中学习并输出预测结果。
