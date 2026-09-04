"""
梯度竞争诊断

监控PINN训练过程中PDE损失和边界损失的梯度方向和大小
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
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

def compute_pde_residual(model, x):
    x = x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    f = torch.sin(np.pi * x)
    residual = -u_xx - f
    return residual

def compute_loss(model, x_interior, x_boundary, rho_bc=100.0):
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)
    total_loss = loss_pde + rho_bc * loss_bc
    return total_loss, loss_pde, loss_bc

def diagnose_gradient_conflict(model, x_interior, x_boundary):
    """
    诊断梯度竞争

    返回:
        cos_angle: 两个梯度的夹角余弦值
        grad_ratio: 梯度范数的比值
        grad_pde_norm: PDE梯度的范数
        grad_bc_norm: BC梯度的范数
    """
    # 计算PDE损失及其梯度
    x_int = x_interior.clone().requires_grad_(True)
    u = model(x_int)
    u_x = torch.autograd.grad(outputs=u, inputs=x_int, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_int, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    f = torch.sin(np.pi * x_int)
    residual = -u_xx - f
    loss_pde = torch.mean(residual ** 2)

    # 计算PDE损失对参数的梯度
    grads_pde = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, allow_unused=True)
    grad_pde = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel())
                          for g, p in zip(grads_pde, model.parameters())])

    # 计算BC损失及其梯度
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)

    grads_bc = torch.autograd.grad(loss_bc, model.parameters(), retain_graph=True, allow_unused=True)
    grad_bc = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel())
                         for g, p in zip(grads_bc, model.parameters())])

    # 计算夹角余弦
    cos_angle = torch.dot(grad_pde, grad_bc) / (grad_pde.norm() * grad_bc.norm() + 1e-8)
    grad_ratio = grad_pde.norm() / (grad_bc.norm() + 1e-8)

    return cos_angle.item(), grad_ratio.item(), grad_pde.norm().item(), grad_bc.norm().item()

if __name__ == "__main__":
    torch.manual_seed(42)
    model = PINN([1, 50, 50, 50, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    history = {
        'cos_angle': [], 'grad_ratio': [],
        'grad_pde': [], 'grad_bc': [],
        'loss_pde': [], 'loss_bc': []
    }

    print("Training and diagnosing gradient conflict...")
    for epoch in range(5000):
        x_interior = torch.rand(100, 1)

        # 诊断梯度（每50个epoch）
        if epoch % 50 == 0:
            cos_angle, grad_ratio, grad_pde, grad_bc = diagnose_gradient_conflict(
                model, x_interior.clone(), x_boundary.clone())
            history['cos_angle'].append(cos_angle)
            history['grad_ratio'].append(grad_ratio)
            history['grad_pde'].append(grad_pde)
            history['grad_bc'].append(grad_bc)

            if cos_angle < -0.5:
                print(f"  Epoch {epoch}: cos(angle) = {cos_angle:.3f} - Severe conflict!")

        # 正常训练
        loss, loss_pde, loss_bc = compute_loss(model, x_interior, x_boundary)

        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: PDE={loss_pde.item():.2e}, BC={loss_bc.item():.2e}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs_diag = list(range(0, 5000, 50))

    # 梯度夹角余弦
    axes[0, 0].plot(epochs_diag, history['cos_angle'], linewidth=1.5, color='purple')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Orthogonal')
    axes[0, 0].fill_between(epochs_diag, -1, 0, alpha=0.2, color='red', label='Conflict region')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('cos(angle)')
    axes[0, 0].set_title('Gradient angle cosine')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-1, 1])

    # 梯度范数
    axes[0, 1].semilogy(epochs_diag, history['grad_pde'], label='PDE gradient', linewidth=1.5)
    axes[0, 1].semilogy(epochs_diag, history['grad_bc'], label='BC gradient', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient norm')
    axes[0, 1].legend()
    axes[0, 1].set_title('Gradient magnitudes')
    axes[0, 1].grid(True, alpha=0.3)

    # 梯度比值
    axes[1, 0].plot(epochs_diag, history['grad_ratio'], linewidth=1.5, color='green')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Balanced')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('|grad_PDE| / |grad_BC|')
    axes[1, 0].set_title('Gradient ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 损失曲线
    axes[1, 1].semilogy(history['loss_pde'], label='PDE loss', linewidth=1.5, alpha=0.7)
    axes[1, 1].semilogy(history['loss_bc'], label='BC loss', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].set_title('Training losses')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_diagnosis.png', dpi=150)
    plt.show()

    # 统计梯度冲突情况
    conflict_count = sum(1 for c in history['cos_angle'] if c < 0)
    print(f"\nGradient conflict episodes: {conflict_count}/{len(history['cos_angle'])} "
          f"({100*conflict_count/len(history['cos_angle']):.1f}%)")
