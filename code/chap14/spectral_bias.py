"""
频谱偏差（Spectral Bias）演示

目标函数：u(x) = sin(2πx) + 0.1*sin(20πx)
包含低频成分和高频成分，观察神经网络的学习行为
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 目标函数：低频 + 高频
def target_function(x):
    return torch.sin(2 * np.pi * x) + 0.1 * torch.sin(20 * np.pi * x)

# 简单的MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    torch.manual_seed(42)

    # 训练并记录不同epoch的结果
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_train = torch.linspace(0, 1, 200).reshape(-1, 1)
    y_train = target_function(x_train)

    checkpoints = [100, 500, 1000, 5000, 10000]
    results = {}
    loss_history = []

    for epoch in range(10001):
        pred = model(x_train)
        loss = torch.mean((pred - y_train)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch in checkpoints:
            with torch.no_grad():
                results[epoch] = model(x_train).numpy().flatten()
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x_np = x_train.numpy().flatten()
    y_np = y_train.numpy().flatten()

    # 上排：不同epoch的拟合结果
    for i, epoch in enumerate(checkpoints[:3]):
        axes[0, i].plot(x_np, y_np, 'b-', label='Target', alpha=0.7, linewidth=2)
        axes[0, i].plot(x_np, results[epoch], 'r--', label='Prediction', linewidth=2)
        axes[0, i].set_title(f'Epoch {epoch}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim([-1.3, 1.3])

    # 下排左和中：后两个checkpoint
    for i, epoch in enumerate(checkpoints[3:5]):
        axes[1, i].plot(x_np, y_np, 'b-', label='Target', alpha=0.7, linewidth=2)
        axes[1, i].plot(x_np, results[epoch], 'r--', label='Prediction', linewidth=2)
        axes[1, i].set_title(f'Epoch {epoch}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim([-1.3, 1.3])

    # 下排右：损失曲线
    axes[1, 2].semilogy(loss_history, linewidth=1.5)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MSE Loss')
    axes[1, 2].set_title('Training loss curve')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_bias.png', dpi=150)
    plt.show()

    print(f"\nFinal MSE: {loss_history[-1]:.6f}")
