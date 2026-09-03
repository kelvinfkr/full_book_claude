"""
重新生成所有PINN实验图片，使用中文标签
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 确保输出目录存在
os.makedirs('figures_chap13', exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("开始生成所有中文标签图片")
print("=" * 60)

# ============================================================
# 实验1: 1D泊松方程PINN
# ============================================================
print("\n[1/8] 1D泊松方程PINN...")

class PINN_1D(nn.Module):
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
        return self.net(x)

def exact_solution_1d(x):
    return torch.sin(np.pi * x)

def train_pinn_1d():
    model = PINN_1D(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_interior = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    x_boundary = torch.tensor([[0.0], [1.0]])

    losses = []
    for epoch in range(2000):
        optimizer.zero_grad()

        # PDE残差
        u = model(x_interior)
        u_x = torch.autograd.grad(u, x_interior, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_interior, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_interior)
        pde_residual = u_xx + f

        # 边界条件
        u_bc = model(x_boundary)

        loss_pde = torch.mean(pde_residual ** 2)
        loss_bc = torch.mean(u_bc ** 2)
        loss = loss_pde + 100 * loss_bc

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

model_1d, losses_1d = train_pinn_1d()

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 损失曲线
ax1 = axes[0]
ax1.semilogy(losses_1d)
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('损失值', fontsize=13, labelpad=10)
ax1.set_title('训练损失曲线', fontsize=14, pad=15)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 解对比
ax2 = axes[1]
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model_1d(x_test).numpy()
u_exact = exact_solution_1d(x_test).numpy()

ax2.plot(x_test.numpy(), u_exact, 'b-', linewidth=2, label='精确解')
ax2.plot(x_test.numpy(), u_pred, 'r--', linewidth=2, label='PINN预测')
ax2.set_xlabel('$x$', fontsize=13, labelpad=10)
ax2.set_ylabel('$u(x)$', fontsize=13, labelpad=10)
ax2.set_title('PINN解与精确解对比', fontsize=14, pad=15)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/pinn_poisson_1d.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/pinn_poisson_1d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 pinn_poisson_1d.pdf/png")

# ============================================================
# 实验2: 频谱偏置
# ============================================================
print("\n[2/8] 频谱偏置演示...")

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

def train_spectral_bias():
    # 目标函数：低频 + 高频
    def target_func(x):
        return torch.sin(2 * np.pi * x) + 0.5 * torch.sin(20 * np.pi * x)

    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.linspace(0, 1, 200).reshape(-1, 1)
    y_train = target_func(x_train)

    history = {'epochs': [], 'predictions': []}

    for epoch in range(5001):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = torch.mean((y_pred - y_train) ** 2)
        loss.backward()
        optimizer.step()

        if epoch in [0, 500, 2000, 5000]:
            with torch.no_grad():
                history['epochs'].append(epoch)
                history['predictions'].append(model(x_train).numpy().copy())

    return model, history, x_train, y_train

model_sb, history_sb, x_sb, y_sb = train_spectral_bias()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
epochs_to_show = [0, 500, 2000, 5000]
titles = ['初始状态 (第0轮)', '第500轮', '第2000轮', '第5000轮']

for idx, (ax, epoch, title) in enumerate(zip(axes.flat, epochs_to_show, titles)):
    ax.plot(x_sb.numpy(), y_sb.numpy(), 'b-', linewidth=2, label='目标函数', alpha=0.7)
    ax.plot(x_sb.numpy(), history_sb['predictions'][idx], 'r--', linewidth=2, label='网络输出')
    ax.set_xlabel('$x$', fontsize=12, labelpad=8)
    ax.set_ylabel('$y$', fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10, pad=4)
    ax.set_ylim([-2, 2])

plt.suptitle('频谱偏置现象：网络先学会低频成分', fontsize=15, y=1.02)
plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/spectral_bias.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/spectral_bias.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 spectral_bias.pdf/png")

# ============================================================
# 实验3: 傅里叶特征
# ============================================================
print("\n[3/8] 傅里叶特征对比...")

class FourierFeatureNet(nn.Module):
    def __init__(self, num_frequencies=10, sigma=10.0):
        super().__init__()
        self.B = torch.randn(1, num_frequencies) * sigma
        self.net = nn.Sequential(
            nn.Linear(2 * num_frequencies, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_proj = x @ self.B
        features = torch.cat([torch.sin(2 * np.pi * x_proj),
                              torch.cos(2 * np.pi * x_proj)], dim=-1)
        return self.net(features)

def target_func_ff(x):
    return torch.sin(2 * np.pi * x) + 0.5 * torch.sin(20 * np.pi * x)

# 训练标准MLP
mlp = SimpleMLP()
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
x_train_ff = torch.linspace(0, 1, 200).reshape(-1, 1)
y_train_ff = target_func_ff(x_train_ff)

losses_mlp = []
for epoch in range(3000):
    optimizer_mlp.zero_grad()
    loss = torch.mean((mlp(x_train_ff) - y_train_ff) ** 2)
    loss.backward()
    optimizer_mlp.step()
    losses_mlp.append(loss.item())

# 训练傅里叶特征网络
ff_net = FourierFeatureNet(num_frequencies=20, sigma=10.0)
optimizer_ff = torch.optim.Adam(ff_net.parameters(), lr=0.001)

losses_ff = []
for epoch in range(3000):
    optimizer_ff.zero_grad()
    loss = torch.mean((ff_net(x_train_ff) - y_train_ff) ** 2)
    loss.backward()
    optimizer_ff.step()
    losses_ff.append(loss.item())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 损失对比
ax1 = axes[0]
ax1.semilogy(losses_mlp, label='标准MLP', linewidth=2)
ax1.semilogy(losses_ff, label='傅里叶特征网络', linewidth=2)
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('损失值', fontsize=13, labelpad=10)
ax1.set_title('训练损失对比', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 预测对比
ax2 = axes[1]
with torch.no_grad():
    y_mlp = mlp(x_train_ff).numpy()
    y_ff = ff_net(x_train_ff).numpy()

ax2.plot(x_train_ff.numpy(), y_train_ff.numpy(), 'b-', linewidth=2, label='目标函数', alpha=0.7)
ax2.plot(x_train_ff.numpy(), y_mlp, 'g--', linewidth=2, label='标准MLP')
ax2.plot(x_train_ff.numpy(), y_ff, 'r:', linewidth=2, label='傅里叶特征')
ax2.set_xlabel('$x$', fontsize=13, labelpad=10)
ax2.set_ylabel('$y$', fontsize=13, labelpad=10)
ax2.set_title('拟合效果对比', fontsize=14, pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/fourier_features.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/fourier_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 fourier_features.pdf/png")

# ============================================================
# 实验4: 硬约束 vs 软约束
# ============================================================
print("\n[4/8] 硬约束与软约束对比...")

class SoftConstraintPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class HardConstraintPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 硬约束: u(x) = x(1-x) * NN(x)
        # 自动满足 u(0) = u(1) = 0
        return x * (1 - x) * self.net(x)

def train_constraint_comparison():
    # 训练软约束
    soft_model = SoftConstraintPINN()
    soft_opt = torch.optim.Adam(soft_model.parameters(), lr=0.001)

    x_int = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    x_bc = torch.tensor([[0.0], [1.0]])

    soft_losses = []
    for epoch in range(2000):
        soft_opt.zero_grad()

        u = soft_model(x_int)
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_int)

        loss_pde = torch.mean((u_xx + f) ** 2)
        loss_bc = torch.mean(soft_model(x_bc) ** 2)
        loss = loss_pde + 100 * loss_bc

        loss.backward()
        soft_opt.step()
        soft_losses.append(loss.item())

    # 训练硬约束
    hard_model = HardConstraintPINN()
    hard_opt = torch.optim.Adam(hard_model.parameters(), lr=0.001)

    hard_losses = []
    for epoch in range(2000):
        hard_opt.zero_grad()

        x_int_h = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
        u = hard_model(x_int_h)
        u_x = torch.autograd.grad(u, x_int_h, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int_h, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_int_h)

        loss = torch.mean((u_xx + f) ** 2)

        loss.backward()
        hard_opt.step()
        hard_losses.append(loss.item())

    return soft_model, hard_model, soft_losses, hard_losses

soft_m, hard_m, soft_l, hard_l = train_constraint_comparison()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 损失对比
ax1 = axes[0]
ax1.semilogy(soft_l, label='软约束 (惩罚法)', linewidth=2)
ax1.semilogy(hard_l, label='硬约束 (结构设计)', linewidth=2)
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('损失值', fontsize=13, labelpad=10)
ax1.set_title('训练损失对比', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 解对比
ax2 = axes[1]
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_exact = torch.sin(np.pi * x_test).numpy()

with torch.no_grad():
    u_soft = soft_m(x_test).numpy()
    u_hard = hard_m(x_test).numpy()

ax2.plot(x_test.numpy(), u_exact, 'b-', linewidth=2.5, label='精确解')
ax2.plot(x_test.numpy(), u_soft, 'g--', linewidth=2, label='软约束')
ax2.plot(x_test.numpy(), u_hard, 'r:', linewidth=2, label='硬约束')
ax2.set_xlabel('$x$', fontsize=13, labelpad=10)
ax2.set_ylabel('$u(x)$', fontsize=13, labelpad=10)
ax2.set_title('解的对比', fontsize=14, pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/hard_vs_soft.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/hard_vs_soft.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 hard_vs_soft.pdf/png")

# ============================================================
# 实验5: 自适应权重
# ============================================================
print("\n[5/8] 自适应权重训练...")

def train_adaptive_weight():
    model = SoftConstraintPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 可学习的拉格朗日乘子
    log_lambda = torch.tensor([0.0], requires_grad=True)
    lambda_optimizer = torch.optim.Adam([log_lambda], lr=0.01)

    x_int = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])

    losses_pde = []
    losses_bc = []
    lambdas = []

    for epoch in range(2000):
        x_int_g = x_int.clone().requires_grad_(True)

        # 前向传播
        u = model(x_int_g)
        u_x = torch.autograd.grad(u, x_int_g, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int_g, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_int_g)

        loss_pde = torch.mean((u_xx + f) ** 2)
        loss_bc = torch.mean(model(x_bc) ** 2)

        lam = torch.exp(log_lambda)

        # 更新网络参数 (最小化)
        optimizer.zero_grad()
        loss = loss_pde + lam.detach() * loss_bc
        loss.backward()
        optimizer.step()

        # 更新拉格朗日乘子 (最大化)
        x_int_g2 = x_int.clone().requires_grad_(True)
        u2 = model(x_int_g2)
        u_x2 = torch.autograd.grad(u2, x_int_g2, torch.ones_like(u2), create_graph=True)[0]
        u_xx2 = torch.autograd.grad(u_x2, x_int_g2, torch.ones_like(u_x2), create_graph=True)[0]

        loss_pde2 = torch.mean((u_xx2 + f.detach()) ** 2)
        loss_bc2 = torch.mean(model(x_bc) ** 2)

        lambda_optimizer.zero_grad()
        dual_loss = -(loss_pde2 + lam * loss_bc2)
        dual_loss.backward()
        lambda_optimizer.step()

        losses_pde.append(loss_pde.item())
        losses_bc.append(loss_bc.item())
        lambdas.append(lam.item())

    return model, losses_pde, losses_bc, lambdas

ada_model, ada_pde, ada_bc, ada_lam = train_adaptive_weight()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax1 = axes[0]
ax1.semilogy(ada_pde, label='PDE损失', linewidth=2)
ax1.semilogy(ada_bc, label='边界条件损失', linewidth=2)
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('损失值', fontsize=13, labelpad=10)
ax1.set_title('各项损失变化', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

ax2 = axes[1]
ax2.plot(ada_lam, 'purple', linewidth=2)
ax2.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax2.set_ylabel('$\\lambda$', fontsize=13, labelpad=10)
ax2.set_title('自适应权重 $\\lambda$ 变化', fontsize=14, pad=15)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

ax3 = axes[2]
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_exact = torch.sin(np.pi * x_test).numpy()
with torch.no_grad():
    u_ada = ada_model(x_test).numpy()

ax3.plot(x_test.numpy(), u_exact, 'b-', linewidth=2.5, label='精确解')
ax3.plot(x_test.numpy(), u_ada, 'r--', linewidth=2, label='自适应权重PINN')
ax3.set_xlabel('$x$', fontsize=13, labelpad=10)
ax3.set_ylabel('$u(x)$', fontsize=13, labelpad=10)
ax3.set_title('解的对比', fontsize=14, pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/adaptive_weight.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/adaptive_weight.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 adaptive_weight.pdf/png")

# ============================================================
# 实验6: 梯度诊断
# ============================================================
print("\n[6/8] 梯度冲突诊断...")

def diagnose_gradient_conflict():
    model = SoftConstraintPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_int = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])

    cosine_similarities = []
    grad_norms_pde = []
    grad_norms_bc = []

    # 获取参数列表以确保一致性
    params = list(model.parameters())

    for epoch in range(1000):
        x_int_g = x_int.clone().requires_grad_(True)

        u = model(x_int_g)
        u_x = torch.autograd.grad(u, x_int_g, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int_g, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_int_g)

        loss_pde = torch.mean((u_xx + f) ** 2)
        loss_bc = torch.mean(model(x_bc) ** 2)

        # 计算各自梯度 - 使用相同的参数列表
        grads_pde = torch.autograd.grad(loss_pde, params, retain_graph=True, allow_unused=True)
        grads_bc = torch.autograd.grad(loss_bc, params, retain_graph=True, allow_unused=True)

        # 展平并计算 - 确保两边使用相同的参数集
        grad_pde_list = []
        grad_bc_list = []
        for g_pde, g_bc in zip(grads_pde, grads_bc):
            if g_pde is not None and g_bc is not None:
                grad_pde_list.append(g_pde.flatten())
                grad_bc_list.append(g_bc.flatten())

        if grad_pde_list and grad_bc_list:
            grad_pde_flat = torch.cat(grad_pde_list)
            grad_bc_flat = torch.cat(grad_bc_list)

            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                grad_pde_flat.unsqueeze(0), grad_bc_flat.unsqueeze(0)
            ).item()

            cosine_similarities.append(cos_sim)
            grad_norms_pde.append(torch.norm(grad_pde_flat).item())
            grad_norms_bc.append(torch.norm(grad_bc_flat).item())
        else:
            cosine_similarities.append(0.0)
            grad_norms_pde.append(0.0)
            grad_norms_bc.append(0.0)

        # 正常训练
        optimizer.zero_grad()
        loss = loss_pde + 100 * loss_bc
        loss.backward()
        optimizer.step()

    return cosine_similarities, grad_norms_pde, grad_norms_bc

cos_sims, gnorms_pde, gnorms_bc = diagnose_gradient_conflict()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax1 = axes[0]
ax1.plot(cos_sims, 'b-', linewidth=1.5, alpha=0.7)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='正交 (cos=0)')
ax1.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax1.set_ylabel('余弦相似度', fontsize=13, labelpad=10)
ax1.set_title('PDE梯度与边界条件梯度的夹角', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-1.1, 1.1])
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

ax2 = axes[1]
ax2.semilogy(gnorms_pde, label='PDE梯度范数', linewidth=2)
ax2.semilogy(gnorms_bc, label='边界条件梯度范数', linewidth=2)
ax2.set_xlabel('训练轮数', fontsize=13, labelpad=10)
ax2.set_ylabel('梯度范数', fontsize=13, labelpad=10)
ax2.set_title('各项损失的梯度范数', fontsize=14, pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/gradient_diagnosis.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/gradient_diagnosis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 gradient_diagnosis.pdf/png")

# ============================================================
# 实验7: Lambda效应
# ============================================================
print("\n[7/8] Lambda惩罚系数效应...")

def train_with_lambda(lam_value, epochs=2000):
    model = SoftConstraintPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_int = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])

    for epoch in range(epochs):
        x_int_g = x_int.clone().requires_grad_(True)

        u = model(x_int_g)
        u_x = torch.autograd.grad(u, x_int_g, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int_g, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * x_int_g)

        loss_pde = torch.mean((u_xx + f) ** 2)
        loss_bc = torch.mean(model(x_bc) ** 2)
        loss = loss_pde + lam_value * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

lambdas_to_test = [0.1, 1.0, 10.0, 100.0, 1000.0]
models_lam = {lam: train_with_lambda(lam) for lam in lambdas_to_test}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_exact = torch.sin(np.pi * x_test).numpy()

# 全局对比
ax1 = axes[0]
ax1.plot(x_test.numpy(), u_exact, 'k-', linewidth=3, label='精确解')
colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas_to_test)))
for lam, color in zip(lambdas_to_test, colors):
    with torch.no_grad():
        u_pred = models_lam[lam](x_test).numpy()
    ax1.plot(x_test.numpy(), u_pred, '--', color=color, linewidth=2, label=f'$\\lambda$={lam}')
ax1.set_xlabel('$x$', fontsize=13, labelpad=10)
ax1.set_ylabel('$u(x)$', fontsize=13, labelpad=10)
ax1.set_title('不同$\\lambda$值的预测结果', fontsize=14, pad=15)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# 边界误差对比
ax2 = axes[1]
bc_errors = []
pde_errors = []
for lam in lambdas_to_test:
    with torch.no_grad():
        bc_err = abs(models_lam[lam](torch.tensor([[0.0]])).item()) + \
                 abs(models_lam[lam](torch.tensor([[1.0]])).item())
        u_pred = models_lam[lam](x_test).numpy()
        pde_err = np.mean((u_pred - u_exact) ** 2)
    bc_errors.append(bc_err)
    pde_errors.append(pde_err)

x_pos = np.arange(len(lambdas_to_test))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, bc_errors, width, label='边界误差', color='steelblue')
bars2 = ax2.bar(x_pos + width/2, pde_errors, width, label='整体MSE', color='coral')
ax2.set_xlabel('$\\lambda$值', fontsize=13, labelpad=10)
ax2.set_ylabel('误差', fontsize=13, labelpad=10)
ax2.set_title('不同$\\lambda$值的误差分析', fontsize=14, pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(l) for l in lambdas_to_test])
ax2.legend(fontsize=11)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/lambda_effect.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/lambda_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 lambda_effect.pdf/png")

# ============================================================
# 实验8: 2D泊松方程
# ============================================================
print("\n[8/8] 2D泊松方程PINN...")

class PINN_2D(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, xy):
        # 硬约束: u = x(1-x)y(1-y) * NN
        x, y = xy[:, 0:1], xy[:, 1:2]
        boundary_factor = x * (1 - x) * y * (1 - y)
        return boundary_factor * self.net(xy)

def exact_solution_2d(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def train_pinn_2d():
    model = PINN_2D(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 内部采样点
    n_points = 50
    x = torch.linspace(0, 1, n_points)
    y = torch.linspace(0, 1, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy_interior = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)

    losses = []
    for epoch in range(3000):
        optimizer.zero_grad()

        u = model(xy_interior)

        # 计算拉普拉斯算子
        grad_u = torch.autograd.grad(u, xy_interior, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        u_xx = torch.autograd.grad(u_x, xy_interior, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_interior, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        laplacian = u_xx + u_yy

        # 源项
        x_pts = xy_interior[:, 0:1]
        y_pts = xy_interior[:, 1:2]
        f = 2 * (np.pi ** 2) * torch.sin(np.pi * x_pts) * torch.sin(np.pi * y_pts)

        loss = torch.mean((laplacian + f) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 500 == 0:
            print(f"    轮次 {epoch}, 损失: {loss.item():.6f}")

    return model, losses

model_2d, losses_2d = train_pinn_2d()

# 生成网格用于可视化
n_viz = 50
x_viz = np.linspace(0, 1, n_viz)
y_viz = np.linspace(0, 1, n_viz)
X_viz, Y_viz = np.meshgrid(x_viz, y_viz)

# 精确解
U_exact = exact_solution_2d(X_viz, Y_viz)

# PINN预测
xy_viz = torch.tensor(np.stack([X_viz.flatten(), Y_viz.flatten()], axis=1), dtype=torch.float32)
with torch.no_grad():
    U_pred = model_2d(xy_viz).numpy().reshape(n_viz, n_viz)

# 误差
U_error = np.abs(U_pred - U_exact)

# 绘制主图
fig = plt.figure(figsize=(15, 10))

# 3D精确解
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X_viz, Y_viz, U_exact, cmap='viridis', alpha=0.8)
ax1.set_xlabel('$x$', fontsize=12, labelpad=10)
ax1.set_ylabel('$y$', fontsize=12, labelpad=10)
ax1.set_zlabel('$u$', fontsize=12, labelpad=10)
ax1.set_title('精确解', fontsize=14, pad=15)
ax1.tick_params(axis='both', which='major', labelsize=9, pad=3)

# 3D PINN预测
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X_viz, Y_viz, U_pred, cmap='viridis', alpha=0.8)
ax2.set_xlabel('$x$', fontsize=12, labelpad=10)
ax2.set_ylabel('$y$', fontsize=12, labelpad=10)
ax2.set_zlabel('$u$', fontsize=12, labelpad=10)
ax2.set_title('PINN预测解', fontsize=14, pad=15)
ax2.tick_params(axis='both', which='major', labelsize=9, pad=3)

# 3D误差
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X_viz, Y_viz, U_error, cmap='hot', alpha=0.8)
ax3.set_xlabel('$x$', fontsize=12, labelpad=10)
ax3.set_ylabel('$y$', fontsize=12, labelpad=10)
ax3.set_zlabel('误差', fontsize=12, labelpad=10)
ax3.set_title('绝对误差', fontsize=14, pad=15)
ax3.tick_params(axis='both', which='major', labelsize=9, pad=3)

# 等高线 - 精确解
ax4 = fig.add_subplot(2, 3, 4)
c1 = ax4.contourf(X_viz, Y_viz, U_exact, levels=20, cmap='viridis')
plt.colorbar(c1, ax=ax4, shrink=0.8)
ax4.set_xlabel('$x$', fontsize=12, labelpad=8)
ax4.set_ylabel('$y$', fontsize=12, labelpad=8)
ax4.set_title('精确解等高线', fontsize=14, pad=12)
ax4.set_aspect('equal')
ax4.tick_params(axis='both', which='major', labelsize=10, pad=4)

# 等高线 - PINN
ax5 = fig.add_subplot(2, 3, 5)
c2 = ax5.contourf(X_viz, Y_viz, U_pred, levels=20, cmap='viridis')
plt.colorbar(c2, ax=ax5, shrink=0.8)
ax5.set_xlabel('$x$', fontsize=12, labelpad=8)
ax5.set_ylabel('$y$', fontsize=12, labelpad=8)
ax5.set_title('PINN预测等高线', fontsize=14, pad=12)
ax5.set_aspect('equal')
ax5.tick_params(axis='both', which='major', labelsize=10, pad=4)

# 训练损失
ax6 = fig.add_subplot(2, 3, 6)
ax6.semilogy(losses_2d)
ax6.set_xlabel('训练轮数', fontsize=12, labelpad=8)
ax6.set_ylabel('损失值', fontsize=12, labelpad=8)
ax6.set_title('训练损失曲线', fontsize=14, pad=12)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='both', which='major', labelsize=10, pad=4)

plt.tight_layout(pad=2.5)
plt.savefig('figures_chap13/pinn_poisson_2d.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/pinn_poisson_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 pinn_poisson_2d.pdf/png")

# 切片对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# x=0.5 切片
ax1 = axes[0]
mid_idx = n_viz // 2
ax1.plot(y_viz, U_exact[mid_idx, :], 'b-', linewidth=2.5, label='精确解')
ax1.plot(y_viz, U_pred[mid_idx, :], 'r--', linewidth=2, label='PINN预测')
ax1.set_xlabel('$y$', fontsize=13, labelpad=10)
ax1.set_ylabel('$u(0.5, y)$', fontsize=13, labelpad=10)
ax1.set_title('$x=0.5$ 截面对比', fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=11, pad=5)

# y=0.5 切片
ax2 = axes[1]
ax2.plot(x_viz, U_exact[:, mid_idx], 'b-', linewidth=2.5, label='精确解')
ax2.plot(x_viz, U_pred[:, mid_idx], 'r--', linewidth=2, label='PINN预测')
ax2.set_xlabel('$x$', fontsize=13, labelpad=10)
ax2.set_ylabel('$u(x, 0.5)$', fontsize=13, labelpad=10)
ax2.set_title('$y=0.5$ 截面对比', fontsize=14, pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=11, pad=5)

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/pinn_poisson_2d_slices.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/pinn_poisson_2d_slices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 pinn_poisson_2d_slices.pdf/png")

# ============================================================
# 额外图：PINN架构示意图（针对大二学生）
# ============================================================
print("\n[额外] 生成PINN概念示意图...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# 绘制PINN流程图
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# 输入层
rect1 = plt.Rectangle((0.5, 3), 2, 2, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(rect1)
ax.text(1.5, 4, '输入\n$(x, t)$', ha='center', va='center', fontsize=12, fontweight='bold')

# 神经网络
rect2 = plt.Rectangle((4, 2.5), 3, 3, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2)
ax.add_patch(rect2)
ax.text(5.5, 4, '神经网络\n$u_\\theta(x,t)$', ha='center', va='center', fontsize=12, fontweight='bold')

# 自动微分
rect3 = plt.Rectangle((8.5, 2.5), 3, 3, fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2)
ax.add_patch(rect3)
ax.text(10, 4, '自动微分\n$\\frac{\\partial u}{\\partial t}, \\frac{\\partial^2 u}{\\partial x^2}$',
        ha='center', va='center', fontsize=11, fontweight='bold')

# 箭头
ax.annotate('', xy=(3.9, 4), xytext=(2.6, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(8.4, 4), xytext=(7.1, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# 损失函数框
rect_pde = plt.Rectangle((9, 0.3), 2.5, 1.2, fill=True, facecolor='#ffcccc', edgecolor='red', linewidth=2)
ax.add_patch(rect_pde)
ax.text(10.25, 0.9, 'PDE损失\n$\\mathcal{L}_{pde}$', ha='center', va='center', fontsize=10, fontweight='bold')

rect_bc = plt.Rectangle((6, 0.3), 2.5, 1.2, fill=True, facecolor='#ccccff', edgecolor='blue', linewidth=2)
ax.add_patch(rect_bc)
ax.text(7.25, 0.9, '边界损失\n$\\mathcal{L}_{bc}$', ha='center', va='center', fontsize=10, fontweight='bold')

rect_ic = plt.Rectangle((3, 0.3), 2.5, 1.2, fill=True, facecolor='#ccffcc', edgecolor='green', linewidth=2)
ax.add_patch(rect_ic)
ax.text(4.25, 0.9, '初值损失\n$\\mathcal{L}_{ic}$', ha='center', va='center', fontsize=10, fontweight='bold')

# 总损失
rect_total = plt.Rectangle((5.5, 6.5), 3, 1.2, fill=True, facecolor='#ffcc99', edgecolor='orange', linewidth=2)
ax.add_patch(rect_total)
ax.text(7, 7.1, '总损失 $\\mathcal{L} = \\mathcal{L}_{pde} + \\lambda_{bc}\\mathcal{L}_{bc} + \\lambda_{ic}\\mathcal{L}_{ic}$',
        ha='center', va='center', fontsize=10, fontweight='bold')

# 连接到总损失的箭头
ax.annotate('', xy=(6.5, 6.4), xytext=(4.5, 1.6),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='green', connectionstyle='arc3,rad=0.2'))
ax.annotate('', xy=(7, 6.4), xytext=(7.25, 1.6),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(7.5, 6.4), xytext=(10.25, 1.6),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', connectionstyle='arc3,rad=-0.2'))

# 反向传播
ax.annotate('', xy=(5.5, 5.6), xytext=(6.2, 6.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple', connectionstyle='arc3,rad=0.3'))
ax.text(4.2, 6.2, '反向传播\n更新参数', ha='center', va='center', fontsize=10, color='purple', fontweight='bold')

# 从自动微分到损失
ax.annotate('', xy=(10.25, 1.6), xytext=(10, 2.4),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

ax.set_title('PINN训练流程示意图', fontsize=16, pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('figures_chap13/pinn_workflow.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/pinn_workflow.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 pinn_workflow.pdf/png")

# ============================================================
# 额外图：软约束vs硬约束概念图
# ============================================================
print("\n[额外] 生成软硬约束概念对比图...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 软约束
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')

rect1 = plt.Rectangle((1, 3), 3, 2, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(rect1)
ax1.text(2.5, 4, '神经网络\n$\\text{NN}(x)$', ha='center', va='center', fontsize=11, fontweight='bold')

rect2 = plt.Rectangle((6, 3), 3, 2, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2)
ax1.add_patch(rect2)
ax1.text(7.5, 4, '输出\n$u(x)$', ha='center', va='center', fontsize=11, fontweight='bold')

ax1.annotate('', xy=(5.9, 4), xytext=(4.1, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# 惩罚项
rect3 = plt.Rectangle((3.5, 0.5), 3, 1.5, fill=True, facecolor='#ffcccc', edgecolor='red', linewidth=2)
ax1.add_patch(rect3)
ax1.text(5, 1.25, '惩罚项\n$\\lambda|u(0)-g|^2$', ha='center', va='center', fontsize=10, color='red')

ax1.annotate('', xy=(5, 2.1), xytext=(7.5, 2.9),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', connectionstyle='arc3,rad=-0.3'))

ax1.set_title('软约束：通过惩罚项间接满足边界条件', fontsize=13, pad=15, fontweight='bold')
ax1.text(5, 6.5, '边界条件可能略有误差', ha='center', va='center', fontsize=11, color='red')

# 硬约束
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')

rect1 = plt.Rectangle((0.5, 3), 2.5, 2, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax2.add_patch(rect1)
ax2.text(1.75, 4, '神经网络\n$\\text{NN}(x)$', ha='center', va='center', fontsize=11, fontweight='bold')

rect2 = plt.Rectangle((4, 3), 2.5, 2, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2)
ax2.add_patch(rect2)
ax2.text(5.25, 4, '乘以\n$x(1-x)$', ha='center', va='center', fontsize=11, fontweight='bold', color='green')

rect3 = plt.Rectangle((7.5, 3), 2, 2, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2)
ax2.add_patch(rect3)
ax2.text(8.5, 4, '输出\n$u(x)$', ha='center', va='center', fontsize=11, fontweight='bold')

ax2.annotate('', xy=(3.9, 4), xytext=(3.1, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax2.annotate('', xy=(7.4, 4), xytext=(6.6, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax2.set_title('硬约束：通过结构设计自动满足边界条件', fontsize=13, pad=15, fontweight='bold')
ax2.text(5, 6.5, '$u(0) = 0 \\times (1-0) \\times \\text{NN}(0) = 0$ ✓', ha='center', va='center', fontsize=11, color='green')
ax2.text(5, 5.8, '$u(1) = 1 \\times (1-1) \\times \\text{NN}(1) = 0$ ✓', ha='center', va='center', fontsize=11, color='green')

plt.tight_layout(pad=2.0)
plt.savefig('figures_chap13/constraint_concept.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures_chap13/constraint_concept.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 已保存 constraint_concept.pdf/png")

print("\n" + "=" * 60)
print("所有图片生成完成！")
print("=" * 60)

# 列出生成的文件
import os
files = sorted(os.listdir('figures_chap13'))
print(f"\n共生成 {len(files)} 个文件:")
for f in files:
    print(f"  - {f}")
