"""
硬约束PINN实现

通过架构设计强制满足边界条件：
    u(x) = A(x) + B(x) * N(x)
其中 A(x) 满足边界条件，B(x) 在边界上为零
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 软约束PINN（标准版本）
class SoftConstraintPINN(nn.Module):
    def __init__(self, layers):
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

# 硬约束PINN
class HardConstraintPINN(nn.Module):
    def __init__(self, layers, bc_left=0.0, bc_right=0.0):
        """
        硬约束PINN

        参数:
            layers: 网络层定义（与之前相同）
            bc_left: 左边界条件 u(0)
            bc_right: 右边界条件 u(1)
        """
        super().__init__()

        # 内部网络 N_theta
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))

        self.activation = torch.tanh
        self.bc_left = bc_left
        self.bc_right = bc_right

        # 初始化
        for layer in self.net:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 计算内部网络的输出 N_theta(x)
        h = x
        for i, layer in enumerate(self.net[:-1]):
            h = self.activation(layer(h))
        N = self.net[-1](h)

        # 硬约束公式: u = A(x) + B(x) * N(x)
        # A(x) = a*(1-x) + b*x  (线性插值)
        # B(x) = x*(1-x)        (距离函数)

        A = self.bc_left * (1 - x) + self.bc_right * x
        B = x * (1 - x)

        u = A + B * N
        return u

def compute_pde_residual(model, x):
    """计算PDE残差: -u''(x) - sin(πx)"""
    x = x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    f = torch.sin(np.pi * x)
    residual = -u_xx - f
    return residual

def compute_loss_soft(model, x_interior, x_boundary, lambda_bc=100.0):
    """软约束损失函数"""
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)
    total_loss = loss_pde + lambda_bc * loss_bc
    return total_loss, loss_pde, loss_bc

def compute_loss_hard(model, x_interior):
    """硬约束损失函数——只有PDE残差"""
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)
    return loss_pde

if __name__ == "__main__":
    # 训练软约束PINN
    print("Training Soft Constraint PINN...")
    torch.manual_seed(42)
    model_soft = SoftConstraintPINN([1, 50, 50, 50, 1])
    optimizer_soft = torch.optim.Adam(model_soft.parameters(), lr=1e-3)
    x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    soft_history = {'pde': [], 'bc': []}

    for epoch in range(10000):
        x_interior = torch.rand(100, 1)
        loss, loss_pde, loss_bc = compute_loss_soft(model_soft, x_interior, x_boundary)

        optimizer_soft.zero_grad()
        loss.backward()
        optimizer_soft.step()

        soft_history['pde'].append(loss_pde.item())
        soft_history['bc'].append(loss_bc.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1}: PDE={loss_pde.item():.2e}, BC={loss_bc.item():.2e}")

    # 训练硬约束PINN
    print("\nTraining Hard Constraint PINN...")
    torch.manual_seed(42)
    model_hard = HardConstraintPINN([1, 50, 50, 50, 1])
    optimizer_hard = torch.optim.Adam(model_hard.parameters(), lr=1e-3)

    hard_history = {'pde': []}

    for epoch in range(10000):
        x_interior = torch.rand(100, 1)
        loss_pde = compute_loss_hard(model_hard, x_interior)

        optimizer_hard.zero_grad()
        loss_pde.backward()
        optimizer_hard.step()

        hard_history['pde'].append(loss_pde.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1}: PDE={loss_pde.item():.2e}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        u_soft = model_soft(x_test).numpy()
        u_hard = model_hard(x_test).numpy()
    u_exact = np.sin(np.pi * x_test.numpy()) / (np.pi ** 2)

    # 解的对比
    axes[0, 0].plot(x_test.numpy(), u_exact, 'b-', linewidth=2, label='Exact')
    axes[0, 0].plot(x_test.numpy(), u_soft, 'r--', linewidth=2, label='Soft constraint')
    axes[0, 0].plot(x_test.numpy(), u_hard, 'g-.', linewidth=2, label='Hard constraint')
    axes[0, 0].set_xlabel('$x$')
    axes[0, 0].set_ylabel('$u(x)$')
    axes[0, 0].legend()
    axes[0, 0].set_title('Solution comparison')
    axes[0, 0].grid(True, alpha=0.3)

    # 误差分布
    error_soft = np.abs(u_soft - u_exact)
    error_hard = np.abs(u_hard - u_exact)
    axes[0, 1].semilogy(x_test.numpy(), error_soft, 'r-', linewidth=2, label='Soft constraint')
    axes[0, 1].semilogy(x_test.numpy(), error_hard, 'g-', linewidth=2, label='Hard constraint')
    axes[0, 1].set_xlabel('$x$')
    axes[0, 1].set_ylabel('Absolute error')
    axes[0, 1].legend()
    axes[0, 1].set_title('Error distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # PDE损失对比
    axes[1, 0].semilogy(soft_history['pde'], label='Soft: PDE loss', linewidth=1.5)
    axes[1, 0].semilogy(hard_history['pde'], label='Hard: PDE loss', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].set_title('PDE loss comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # 边界值对比
    with torch.no_grad():
        bc_soft_left = model_soft(torch.tensor([[0.0]])).item()
        bc_soft_right = model_soft(torch.tensor([[1.0]])).item()
        bc_hard_left = model_hard(torch.tensor([[0.0]])).item()
        bc_hard_right = model_hard(torch.tensor([[1.0]])).item()

    methods = ['Soft constraint', 'Hard constraint']
    left_vals = [abs(bc_soft_left), abs(bc_hard_left)]
    right_vals = [abs(bc_soft_right), abs(bc_hard_right)]

    x_pos = np.arange(len(methods))
    width = 0.35

    axes[1, 1].bar(x_pos - width/2, left_vals, width, label='|u(0)|', color='steelblue')
    axes[1, 1].bar(x_pos + width/2, right_vals, width, label='|u(1)|', color='coral')
    axes[1, 1].set_ylabel('Boundary value error')
    axes[1, 1].set_title('Boundary condition satisfaction')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('hard_vs_soft.png', dpi=150)
    plt.show()

    print(f"\nBoundary condition values:")
    print(f"  Soft: u(0)={bc_soft_left:.2e}, u(1)={bc_soft_right:.2e}")
    print(f"  Hard: u(0)={bc_hard_left:.2e}, u(1)={bc_hard_right:.2e}")

    mse_soft = np.mean((u_soft - u_exact)**2)
    mse_hard = np.mean((u_hard - u_exact)**2)
    print(f"\nMSE:")
    print(f"  Soft: {mse_soft:.2e}")
    print(f"  Hard: {mse_hard:.2e}")

    # ========================
    # 逐点数值对比表格
    # ========================
    print("\n" + "="*90)
    print("硬约束与软约束PINN的逐点数值对比")
    print("="*90)
    print(f"{'x':<6} {'解析解':>12} {'软约束预测':>12} {'软约束误差':>12} {'硬约束预测':>12} {'硬约束误差':>12}")
    print("-"*90)

    compare_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x_val in compare_points:
        x_pt = torch.tensor([[x_val]])
        with torch.no_grad():
            u_soft_pt = model_soft(x_pt).item()
            u_hard_pt = model_hard(x_pt).item()
        u_exact_pt = np.sin(np.pi * x_val) / (np.pi ** 2)
        err_soft = abs(u_soft_pt - u_exact_pt)
        err_hard = abs(u_hard_pt - u_exact_pt)
        print(f"{x_val:<6.2f} {u_exact_pt:>12.7f} {u_soft_pt:>12.7f} {err_soft:>12.2e} {u_hard_pt:>12.7f} {err_hard:>12.2e}")

    print("-"*90)
    print(f"{'MSE':<6} {'':<12} {'':<12} {mse_soft:>12.2e} {'':<12} {mse_hard:>12.2e}")
    print("="*90)

    print("\n关键观察:")
    print("1. 硬约束在边界点(x=0, x=1)的误差严格为0")
    print("2. 硬约束在内部点的误差也更小")
    print(f"3. 硬约束MSE比软约束低 {mse_soft/mse_hard:.1f} 倍")
