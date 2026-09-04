"""
傅里叶特征映射（Fourier Features）实验

对比标准MLP和傅里叶特征网络在拟合高频函数上的效果
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def target_function(x):
    return torch.sin(2 * np.pi * x) + 0.1 * torch.sin(20 * np.pi * x)

# 标准MLP
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

# 傅里叶特征映射
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_features, scale=10.0):
        """
        傅里叶特征映射

        参数:
            input_dim: 输入维度
            num_features: 傅里叶特征的数量
            scale: 频率的标准差（控制能学习的最高频率）
        """
        super().__init__()
        # 随机采样频率矩阵（固定，不训练）
        B = torch.randn(num_features, input_dim) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        # x: (batch, input_dim)
        # 计算 2*pi*B*x
        x_proj = 2 * np.pi * x @ self.B.T  # (batch, num_features)
        # 返回 [sin, cos]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# 带傅里叶特征的MLP
class FourierMLP(nn.Module):
    def __init__(self, num_fourier=64, scale=10.0):
        super().__init__()
        self.fourier = FourierFeatures(1, num_fourier, scale)
        self.net = nn.Sequential(
            nn.Linear(2 * num_fourier, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.fourier(x)
        return self.net(h)

if __name__ == "__main__":
    x_train = torch.linspace(0, 1, 200).reshape(-1, 1)
    y_train = target_function(x_train)

    # 比较不同scale的傅里叶特征
    scales = [1.0, 10.0, 30.0]
    fourier_results = {}
    fourier_losses = {}

    for scale in scales:
        print(f"\nScale = {scale}")
        torch.manual_seed(42)
        model = FourierMLP(num_fourier=64, scale=scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss_hist = []
        for epoch in range(5001):
            pred = model(x_train)
            loss = torch.mean((pred - y_train)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

        with torch.no_grad():
            fourier_results[scale] = model(x_train).numpy().flatten()
        fourier_losses[scale] = loss_hist

    # 标准MLP作为对比
    print("\nStandard MLP")
    torch.manual_seed(42)
    model_std = SimpleMLP()
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=1e-3)
    std_loss = []
    for epoch in range(5001):
        pred = model_std(x_train)
        loss = torch.mean((pred - y_train)**2)
        optimizer_std.zero_grad()
        loss.backward()
        optimizer_std.step()
        std_loss.append(loss.item())

    with torch.no_grad():
        std_result = model_std(x_train).numpy().flatten()

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x_np = x_train.numpy().flatten()
    y_np = y_train.numpy().flatten()

    # 标准MLP
    axes[0, 0].plot(x_np, y_np, 'b-', label='Target', alpha=0.7, linewidth=2)
    axes[0, 0].plot(x_np, std_result, 'r--', label='Standard MLP', linewidth=2)
    axes[0, 0].set_title('Standard MLP (5000 epochs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-1.3, 1.3])

    # 傅里叶特征 scale=10
    axes[0, 1].plot(x_np, y_np, 'b-', label='Target', alpha=0.7, linewidth=2)
    axes[0, 1].plot(x_np, fourier_results[10.0], 'r--', label='Fourier (scale=10)', linewidth=2)
    axes[0, 1].set_title('Fourier features (scale=10, 5000 epochs)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-1.3, 1.3])

    # 傅里叶特征 scale=30
    axes[1, 0].plot(x_np, y_np, 'b-', label='Target', alpha=0.7, linewidth=2)
    axes[1, 0].plot(x_np, fourier_results[30.0], 'r--', label='Fourier (scale=30)', linewidth=2)
    axes[1, 0].set_title('Fourier features (scale=30, 5000 epochs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-1.3, 1.3])

    # 损失对比
    axes[1, 1].semilogy(std_loss, label='Standard MLP', linewidth=1.5)
    for scale in scales:
        axes[1, 1].semilogy(fourier_losses[scale], label=f'Fourier (scale={scale})', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('Loss comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fourier_features.png', dpi=150)
    plt.show()

    print(f"\nFinal MSE comparison:")
    print(f"  Standard MLP: {std_loss[-1]:.6f}")
    for scale in scales:
        print(f"  Fourier (scale={scale}): {fourier_losses[scale][-1]:.6f}")
