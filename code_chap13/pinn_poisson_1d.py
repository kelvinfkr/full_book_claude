"""
PINN求解一维泊松方程的完整代码

问题：
    -u''(x) = sin(πx),  x ∈ (0, 1)
    u(0) = 0, u(1) = 0

解析解：u(x) = sin(πx) / π²
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========================
# 第一步：定义神经网络
# ========================
class PINN(nn.Module):
    def __init__(self, layers):
        """
        layers: 每层神经元数量的列表
        例如 [1, 50, 50, 50, 1] 表示输入1维，3个隐藏层各50个神经元，输出1维
        """
        super().__init__()

        # 构建网络层
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # 激活函数
        self.activation = torch.tanh

        # 参数初始化（Xavier初始化，有助于训练稳定性）
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """前向传播"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        # 最后一层不加激活函数
        x = self.layers[-1](x)
        return x

# ========================
# 第二步：计算PDE残差
# ========================
def compute_pde_residual(model, x):
    """
    计算PDE残差: -u''(x) - sin(pi*x)

    使用自动微分计算二阶导数
    """
    # 确保x需要梯度
    x = x.requires_grad_(True)

    # 前向传播得到u
    u = model(x)

    # 计算一阶导数 du/dx
    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,  # 需要继续求导，所以保留计算图
        retain_graph=True
    )[0]

    # 计算二阶导数 d²u/dx²
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # PDE残差: -u'' - f = -u'' - sin(pi*x)
    f = torch.sin(np.pi * x)
    residual = -u_xx - f

    return residual

# ========================
# 第三步：定义损失函数
# ========================
def compute_loss(model, x_interior, x_boundary, lambda_bc=100.0):
    """
    计算PINN总损失

    参数:
        model: 神经网络模型
        x_interior: 内部配点 (N_r, 1)
        x_boundary: 边界点 (N_b, 1)
        lambda_bc: 边界条件的惩罚权重
    """
    # PDE残差损失
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)

    # 边界条件损失: u(0) = 0, u(1) = 0
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)

    # 总损失
    total_loss = loss_pde + lambda_bc * loss_bc

    return total_loss, loss_pde, loss_bc

# ========================
# 第四步：训练循环
# ========================
def train_pinn(model, num_epochs=10000, lr=1e-3, N_interior=100, N_boundary=2):
    """
    训练PINN
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 边界点（固定）
    x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    # 记录损失历史
    history = {'total': [], 'pde': [], 'bc': []}

    for epoch in range(num_epochs):
        # 在每个epoch随机采样内部配点
        x_interior = torch.rand(N_interior, 1)  # 均匀分布在[0,1]

        # 计算损失
        loss, loss_pde, loss_bc = compute_loss(model, x_interior, x_boundary)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        history['total'].append(loss.item())
        history['pde'].append(loss_pde.item())
        history['bc'].append(loss_bc.item())

        # 打印进度
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | "
                  f"PDE: {loss_pde.item():.2e} | BC: {loss_bc.item():.2e}")

    return history

# ========================
# 第五步：主程序
# ========================
if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)

    # 创建网络: 1 -> 50 -> 50 -> 50 -> 1
    model = PINN([1, 50, 50, 50, 1])

    # 训练
    print("开始训练...")
    history = train_pinn(model, num_epochs=10000, lr=1e-3)
    print("训练完成！")

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：解的对比
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).numpy()
    u_exact = np.sin(np.pi * x_test.numpy()) / (np.pi ** 2)

    axes[0].plot(x_test.numpy(), u_exact, 'b-', linewidth=2, label='Exact solution')
    axes[0].plot(x_test.numpy(), u_pred, 'r--', linewidth=2, label='PINN prediction')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$u(x)$')
    axes[0].legend()
    axes[0].set_title('Solution comparison')
    axes[0].grid(True, alpha=0.3)

    # 右图：损失曲线
    axes[1].semilogy(history['total'], label='Total loss')
    axes[1].semilogy(history['pde'], label='PDE loss')
    axes[1].semilogy(history['bc'], label='BC loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_title('Training loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pinn_poisson_1d.png', dpi=150)
    plt.show()

    # 计算误差
    error = np.mean((u_pred - u_exact) ** 2)
    max_error = np.max(np.abs(u_pred - u_exact))
    print(f"\n均方误差 (MSE): {error:.2e}")
    print(f"最大绝对误差: {max_error:.2e}")

    # ========================
    # 数值对比表格
    # ========================
    print("\n" + "="*70)
    print("PINN预测值与解析解的数值对比")
    print("="*70)
    print(f"{'x':<8} {'解析解':>15} {'PINN预测':>15} {'绝对误差':>15}")
    print("-"*70)

    # 选取关键点进行对比
    compare_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for x_val in compare_points:
        x_pt = torch.tensor([[x_val]])
        with torch.no_grad():
            u_pred_pt = model(x_pt).item()
        u_exact_pt = np.sin(np.pi * x_val) / (np.pi ** 2)
        err = abs(u_pred_pt - u_exact_pt)
        print(f"{x_val:<8.2f} {u_exact_pt:>15.7f} {u_pred_pt:>15.7f} {err:>15.2e}")

    print("-"*70)
    print(f"{'MSE':<8} {'':<15} {'':<15} {error:>15.2e}")
    print("="*70)
