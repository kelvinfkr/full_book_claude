"""
额外实验：Deep Ritz Method 和 逆问题 PINN
为没有实验结果的代码部分补充实验
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("生成额外实验图片")
print("=" * 60)

# ============================================================
# 实验1: Deep Ritz Method vs 标准PINN
# ============================================================
print("\n[1/2] Deep Ritz Method vs 标准PINN...")

class SimplePINN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # 硬约束: u(0) = u(1) = 0
        return x * (1 - x) * self.net(x)

def exact_solution(x):
    """精确解: -u'' = pi^2 * sin(pi*x), u(0)=u(1)=0 => u = sin(pi*x)"""
    return torch.sin(np.pi * x)

def source_term(x):
    """源项 f = pi^2 * sin(pi*x)"""
    return (np.pi ** 2) * torch.sin(np.pi * x)

def train_standard_pinn(epochs=2000):
    """标准PINN: 最小化 |u'' + f|^2"""
    model = SimplePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(epochs):
        x = torch.linspace(0.01, 0.99, 50).reshape(-1, 1).requires_grad_(True)

        u = model(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        f = source_term(x)
        # -u'' = f  =>  残差 = u'' + f
        residual = u_xx + f
        loss = torch.mean(residual ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

def train_deep_ritz(epochs=2000):
    """Deep Ritz: 最小化能量泛函 E = (1/2)|∇u|^2 - f*u"""
    model = SimplePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(epochs):
        x = torch.linspace(0.01, 0.99, 50).reshape(-1, 1).requires_grad_(True)

        u = model(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

        f = source_term(x)
        # 能量泛函: E = (1/2)|u'|^2 - f*u (一维情况)
        energy = 0.5 * u_x ** 2 - f * u
        loss = torch.mean(energy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

# 训练两种方法
print("  训练标准PINN...")
pinn_model, pinn_losses = train_standard_pinn(2000)
print("  训练Deep Ritz...")
ritz_model, ritz_losses = train_deep_ritz(2000)

# 评估
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_exact = exact_solution(x_test).numpy()

with torch.no_grad():
    u_pinn = pinn_model(x_test).numpy()
    u_ritz = ritz_model(x_test).numpy()

mse_pinn = np.mean((u_pinn - u_exact) ** 2)
mse_ritz = np.mean((u_ritz - u_exact) ** 2)

print(f"  标准PINN MSE: {mse_pinn:.2e}")
print(f"  Deep Ritz MSE: {mse_ritz:.2e}")

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 解对比
ax1 = axes[0]
ax1.plot(x_test.numpy(), u_exact, 'b-', linewidth=2.5, label='精确解')
ax1.plot(x_test.numpy(), u_pinn, 'g--', linewidth=2, label=f'标准PINN (MSE={mse_pinn:.2e})')
ax1.plot(x_test.numpy(), u_ritz, 'r:', linewidth=2, label=f'Deep Ritz (MSE={mse_ritz:.2e})')
ax1.set_xlabel('$x$', fontsize=13, labelpad=10)
ax1.set_ylabel('$u(x)$', fontsize=13, labelpad=10)
ax1.set_title('解的对比', fontsize=14, pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 损失对比 (标准化以便比较)
ax2 = axes[1]
ax2.semilogy(np.abs(pinn_losses), label='标准PINN (残差)', linewidth=2)
ax2.semilogy(np.abs(ritz_losses), label='Deep Ritz (能量)', linewidth=2)
ax2.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax2.set_ylabel('损失值 (绝对值)', fontsize=13, labelpad=10)
ax2.set_title('训练损失对比', fontsize=14, pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 误差对比
ax3 = axes[2]
error_pinn = np.abs(u_pinn.flatten() - u_exact.flatten())
error_ritz = np.abs(u_ritz.flatten() - u_exact.flatten())
ax3.plot(x_test.numpy(), error_pinn, 'g-', linewidth=2, label='标准PINN误差')
ax3.plot(x_test.numpy(), error_ritz, 'r-', linewidth=2, label='Deep Ritz误差')
ax3.set_xlabel('$x$', fontsize=13, labelpad=10)
ax3.set_ylabel('绝对误差', fontsize=13, labelpad=10)
ax3.set_title('点误差分布', fontsize=14, pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=11, pad=5)
ax3.set_yscale('log')

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap14/deep_ritz_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap14/deep_ritz_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 deep_ritz_comparison.pdf/png")

# ============================================================
# 实验2: 逆问题 - 反演扩散系数
# ============================================================
print("\n[2/2] 逆问题PINN - 反演扩散系数...")

class InversePINN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  # 输入: (x, t)
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # 未知扩散系数 D，初始猜测为0.5
        self.D = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        u_nn = self.net(xt)
        # 硬约束: u(0,t) = u(1,t) = 0 (边界条件)
        return x * (1 - x) * u_nn

def generate_synthetic_data(D_true=0.1, n_obs=50):
    """
    生成合成观测数据
    热方程: u_t = D * u_xx
    初始条件: u(x, 0) = sin(pi*x)
    精确解: u(x, t) = exp(-D*pi^2*t) * sin(pi*x)
    """
    x_obs = torch.rand(n_obs, 1)
    t_obs = torch.rand(n_obs, 1) * 0.5  # t in [0, 0.5]
    u_obs = torch.exp(-D_true * np.pi**2 * t_obs) * torch.sin(np.pi * x_obs)
    # 添加少量噪声
    u_obs = u_obs + 0.01 * torch.randn_like(u_obs)
    return x_obs, t_obs, u_obs

def train_inverse_pinn(D_true=0.1, epochs=5000):
    """训练逆问题PINN，反演扩散系数D"""
    model = InversePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 生成观测数据
    x_obs, t_obs, u_obs = generate_synthetic_data(D_true, n_obs=100)

    D_history = []
    losses = []

    for epoch in range(epochs):
        # 数据拟合损失
        u_pred = model(x_obs, t_obs)
        loss_data = torch.mean((u_pred - u_obs) ** 2)

        # PDE残差损失
        x_coll = torch.rand(100, 1).requires_grad_(True)
        t_coll = torch.rand(100, 1).requires_grad_(True) * 0.5

        u = model(x_coll, t_coll)

        u_t = torch.autograd.grad(u, t_coll, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_coll, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_coll, torch.ones_like(u_x), create_graph=True)[0]

        # PDE残差: u_t - D * u_xx = 0
        residual = u_t - model.D * u_xx
        loss_pde = torch.mean(residual ** 2)

        # 初始条件损失
        x_ic = torch.rand(50, 1).requires_grad_(True)
        t_ic = torch.zeros(50, 1)
        u_ic_pred = model(x_ic, t_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # 总损失
        loss = loss_data + loss_pde + 10 * loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        D_history.append(model.D.item())
        losses.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"    轮次 {epoch+1}: D = {model.D.item():.4f} (真实值: {D_true})")

    return model, D_history, losses, x_obs, t_obs, u_obs

D_true = 0.1
model_inv, D_hist, inv_losses, x_obs, t_obs, u_obs = train_inverse_pinn(D_true, epochs=5000)

D_final = D_hist[-1]
D_error = abs(D_final - D_true) / D_true * 100
print(f"  反演结果: D = {D_final:.4f}")
print(f"  相对误差: {D_error:.2f}%")

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# D的收敛过程
ax1 = axes[0]
ax1.plot(D_hist, 'b-', linewidth=2)
ax1.axhline(y=D_true, color='r', linestyle='--', linewidth=2, label=f'真实值 D={D_true}')
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('反演的 $D$ 值', fontsize=13, labelpad=10)
ax1.set_title('扩散系数反演过程', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 训练损失
ax2 = axes[1]
ax2.semilogy(inv_losses, 'g-', linewidth=2)
ax2.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax2.set_ylabel('总损失', fontsize=13, labelpad=10)
ax2.set_title('训练损失曲线', fontsize=14, pad=15)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 解的对比 (固定t=0.2)
ax3 = axes[2]
x_viz = torch.linspace(0, 1, 100).reshape(-1, 1)
t_viz = torch.full((100, 1), 0.2)

u_exact_viz = torch.exp(-D_true * np.pi**2 * t_viz) * torch.sin(np.pi * x_viz)
with torch.no_grad():
    u_pred_viz = model_inv(x_viz, t_viz)

ax3.plot(x_viz.numpy(), u_exact_viz.numpy(), 'b-', linewidth=2.5, label='精确解')
ax3.plot(x_viz.numpy(), u_pred_viz.numpy(), 'r--', linewidth=2, label='PINN预测')
ax3.scatter(x_obs.numpy(), u_obs.numpy(), c='green', s=20, alpha=0.5, label='观测数据')
ax3.set_xlabel('$x$', fontsize=13, labelpad=10)
ax3.set_ylabel('$u(x, t=0.2)$', fontsize=13, labelpad=10)
ax3.set_title(f'$t=0.2$ 时刻解对比\n(反演 D={D_final:.4f}, 真实 D={D_true})', fontsize=13, pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap14/inverse_problem.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap14/inverse_problem.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 inverse_problem.pdf/png")

print("\n" + "=" * 60)
print("额外实验完成！")
print("=" * 60)

# 输出数值结果供tex使用
print("\n数值结果汇总：")
print(f"Deep Ritz vs 标准PINN:")
print(f"  - 标准PINN MSE: {mse_pinn:.2e}")
print(f"  - Deep Ritz MSE: {mse_ritz:.2e}")
print(f"\n逆问题反演:")
print(f"  - 真实扩散系数: D = {D_true}")
print(f"  - 反演扩散系数: D = {D_final:.4f}")
print(f"  - 相对误差: {D_error:.2f}%")
