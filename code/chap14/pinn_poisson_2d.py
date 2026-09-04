"""
PINN求解二维泊松方程

问题：
    -∇²u = -∂²u/∂x² - ∂²u/∂y² = 2π²sin(πx)sin(πy),  (x,y) ∈ (0,1)²
    u = 0 on ∂Ω (边界)

解析解：u(x,y) = sin(πx)sin(πy)
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================
# 定义神经网络
# ========================
class PINN2D(nn.Module):
    def __init__(self, layers):
        """
        layers: 例如 [2, 64, 64, 64, 1] 表示输入2维(x,y)，输出1维(u)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = torch.tanh
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# ========================
# 硬约束版本（边界自动满足）
# ========================
class HardConstraintPINN2D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = torch.tanh
        for layer in self.net:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]

        # 内部网络输出
        h = xy
        for i, layer in enumerate(self.net[:-1]):
            h = self.activation(layer(h))
        N = self.net[-1](h)

        # 硬约束：u = x(1-x)y(1-y) * N(x,y)
        # 这保证在四条边上 u = 0
        B = x * (1 - x) * y * (1 - y)
        u = B * N
        return u

# ========================
# 计算PDE残差
# ========================
def compute_pde_residual_2d(model, xy):
    """
    计算2D泊松方程残差: -∇²u - f
    f = 2π²sin(πx)sin(πy)
    """
    xy = xy.requires_grad_(True)
    u = model(xy)

    # 计算一阶导数
    grad_u = torch.autograd.grad(
        outputs=u, inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    # 计算二阶导数
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        outputs=u_y, inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]

    # 拉普拉斯算子
    laplacian = u_xx + u_yy

    # 源项 f = 2π²sin(πx)sin(πy)
    x, y = xy[:, 0:1], xy[:, 1:2]
    f = 2 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

    # 残差: -∇²u - f
    residual = -laplacian - f
    return residual

# ========================
# 损失函数
# ========================
def compute_loss_2d(model, xy_interior, xy_boundary, rho_bc=100.0):
    # PDE残差损失
    residual = compute_pde_residual_2d(model, xy_interior)
    loss_pde = torch.mean(residual ** 2)

    # 边界条件损失
    u_boundary = model(xy_boundary)
    loss_bc = torch.mean(u_boundary ** 2)

    total_loss = loss_pde + rho_bc * loss_bc
    return total_loss, loss_pde, loss_bc

def compute_loss_2d_hard(model, xy_interior):
    """硬约束版本只需要PDE损失"""
    residual = compute_pde_residual_2d(model, xy_interior)
    loss_pde = torch.mean(residual ** 2)
    return loss_pde

# ========================
# 采样函数
# ========================
def sample_interior(N):
    """在(0,1)²内部随机采样"""
    xy = torch.rand(N, 2)
    return xy

def sample_boundary(N_per_edge):
    """在边界上采样"""
    # 下边界 y=0
    bottom = torch.stack([torch.rand(N_per_edge), torch.zeros(N_per_edge)], dim=1)
    # 上边界 y=1
    top = torch.stack([torch.rand(N_per_edge), torch.ones(N_per_edge)], dim=1)
    # 左边界 x=0
    left = torch.stack([torch.zeros(N_per_edge), torch.rand(N_per_edge)], dim=1)
    # 右边界 x=1
    right = torch.stack([torch.ones(N_per_edge), torch.rand(N_per_edge)], dim=1)

    return torch.cat([bottom, top, left, right], dim=0)

# ========================
# 解析解
# ========================
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 使用硬约束PINN
    print("="*60)
    print("训练二维泊松方程 PINN（硬约束）")
    print("="*60)

    model = HardConstraintPINN2D([2, 64, 64, 64, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    history = {'loss': []}

    for epoch in range(15000):
        # 采样
        xy_interior = sample_interior(1000)

        # 计算损失
        loss = compute_loss_2d_hard(model, xy_interior)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history['loss'].append(loss.item())

        if (epoch + 1) % 3000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print("训练完成！")

    # ========================
    # 可视化
    # ========================
    # 创建网格
    n_grid = 50
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)

    # 计算解析解
    U_exact = exact_solution(X, Y)

    # 计算PINN预测
    xy_grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(xy_grid).numpy().reshape(n_grid, n_grid)

    # 计算误差
    error = np.abs(U_pred - U_exact)
    mse = np.mean((U_pred - U_exact) ** 2)
    max_error = np.max(error)

    print(f"\nMSE: {mse:.2e}")
    print(f"最大误差: {max_error:.2e}")

    # 绘图
    fig = plt.figure(figsize=(16, 12))

    # 1. 解析解 3D
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_exact, cmap='viridis', alpha=0.9)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$u$')
    ax1.set_title('Exact solution')
    ax1.view_init(elev=25, azim=45)

    # 2. PINN预测 3D
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_pred, cmap='viridis', alpha=0.9)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$u$')
    ax2.set_title('PINN prediction')
    ax2.view_init(elev=25, azim=45)

    # 3. 误差分布 3D
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.9)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_zlabel('Error')
    ax3.set_title(f'Absolute error (max={max_error:.2e})')
    ax3.view_init(elev=25, azim=45)

    # 4. 解析解等高线
    ax4 = fig.add_subplot(2, 3, 4)
    c1 = ax4.contourf(X, Y, U_exact, levels=20, cmap='viridis')
    ax4.set_xlabel('$x$')
    ax4.set_ylabel('$y$')
    ax4.set_title('Exact solution (contour)')
    ax4.set_aspect('equal')
    plt.colorbar(c1, ax=ax4)

    # 5. PINN预测等高线
    ax5 = fig.add_subplot(2, 3, 5)
    c2 = ax5.contourf(X, Y, U_pred, levels=20, cmap='viridis')
    ax5.set_xlabel('$x$')
    ax5.set_ylabel('$y$')
    ax5.set_title('PINN prediction (contour)')
    ax5.set_aspect('equal')
    plt.colorbar(c2, ax=ax5)

    # 6. 误差等高线
    ax6 = fig.add_subplot(2, 3, 6)
    c3 = ax6.contourf(X, Y, error, levels=20, cmap='hot')
    ax6.set_xlabel('$x$')
    ax6.set_ylabel('$y$')
    ax6.set_title(f'Absolute error (MSE={mse:.2e})')
    ax6.set_aspect('equal')
    plt.colorbar(c3, ax=ax6)

    plt.tight_layout()
    plt.savefig('pinn_poisson_2d.png', dpi=150, bbox_inches='tight')
    plt.savefig('pinn_poisson_2d.pdf', dpi=150, bbox_inches='tight')
    plt.show()

    # ========================
    # 切片对比图
    # ========================
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

    # y=0.5 切片
    idx_y = n_grid // 2
    axes[0].plot(x, U_exact[idx_y, :], 'b-', linewidth=2, label='Exact')
    axes[0].plot(x, U_pred[idx_y, :], 'r--', linewidth=2, label='PINN')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$u(x, 0.5)$')
    axes[0].set_title('Slice at $y=0.5$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # x=0.5 切片
    idx_x = n_grid // 2
    axes[1].plot(y, U_exact[:, idx_x], 'b-', linewidth=2, label='Exact')
    axes[1].plot(y, U_pred[:, idx_x], 'r--', linewidth=2, label='PINN')
    axes[1].set_xlabel('$y$')
    axes[1].set_ylabel('$u(0.5, y)$')
    axes[1].set_title('Slice at $x=0.5$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 损失曲线
    axes[2].semilogy(history['loss'], linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pinn_poisson_2d_slices.png', dpi=150, bbox_inches='tight')
    plt.savefig('pinn_poisson_2d_slices.pdf', dpi=150, bbox_inches='tight')
    plt.show()

    # ========================
    # 数值对比表格
    # ========================
    print("\n" + "="*80)
    print("二维泊松方程：PINN预测与解析解的数值对比")
    print("="*80)
    print(f"{'(x, y)':<15} {'解析解':>15} {'PINN预测':>15} {'绝对误差':>15}")
    print("-"*80)

    compare_points = [
        (0.0, 0.0), (0.5, 0.0), (1.0, 0.5),  # 边界点
        (0.25, 0.25), (0.5, 0.5), (0.75, 0.75),  # 内部点
        (0.25, 0.75), (0.75, 0.25)  # 其他内部点
    ]

    for (px, py) in compare_points:
        xy_pt = torch.tensor([[px, py]], dtype=torch.float32)
        with torch.no_grad():
            u_pred_pt = model(xy_pt).item()
        u_exact_pt = exact_solution(px, py)
        err = abs(u_pred_pt - u_exact_pt)
        print(f"({px:.2f}, {py:.2f}){'':<6} {u_exact_pt:>15.7f} {u_pred_pt:>15.7f} {err:>15.2e}")

    print("-"*80)
    print(f"{'MSE':<15} {'':<15} {'':<15} {mse:>15.2e}")
    print("="*80)

    print("\n关键观察:")
    print("1. 边界点的预测值接近0（硬约束保证）")
    print("2. 内部点（如(0.5,0.5)）的误差很小")
    print("3. 解的整体形态与解析解高度一致")
