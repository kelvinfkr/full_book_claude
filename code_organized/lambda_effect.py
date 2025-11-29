"""
罚参数lambda的影响实验

比较不同罚参数值对PINN训练效果的影响
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

def compute_loss(model, x_interior, x_boundary, lambda_bc=100.0):
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)
    total_loss = loss_pde + lambda_bc * loss_bc
    return total_loss, loss_pde, loss_bc

if __name__ == "__main__":
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u_exact = np.sin(np.pi * x_test.numpy()) / (np.pi ** 2)

    lambda_values = [1.0, 10.0, 100.0, 1000.0]
    results = {}

    for lam in lambda_values:
        print(f"\nTraining with λ = {lam}")
        torch.manual_seed(42)
        model = PINN([1, 50, 50, 50, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        history = {'pde': [], 'bc': []}

        for epoch in range(10000):
            x_interior = torch.rand(100, 1)
            loss, loss_pde, loss_bc = compute_loss(model, x_interior, x_boundary, lambda_bc=lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history['pde'].append(loss_pde.item())
            history['bc'].append(loss_bc.item())

        with torch.no_grad():
            u_pred = model(x_test).numpy()

        results[lam] = {
            'history': history,
            'prediction': u_pred,
            'mse': np.mean((u_pred - u_exact)**2),
            'bc_left': model(torch.tensor([[0.0]])).item(),
            'bc_right': model(torch.tensor([[1.0]])).item()
        }

        print(f"  Final PDE loss: {history['pde'][-1]:.2e}")
        print(f"  Final BC loss: {history['bc'][-1]:.2e}")
        print(f"  MSE: {results[lam]['mse']:.2e}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['blue', 'green', 'orange', 'red']

    # 解的对比
    axes[0, 0].plot(x_test.numpy(), u_exact, 'k-', linewidth=2, label='Exact')
    for i, lam in enumerate(lambda_values):
        axes[0, 0].plot(x_test.numpy(), results[lam]['prediction'],
                        linestyle='--', linewidth=1.5, color=colors[i], label=f'$\\lambda$={lam}')
    axes[0, 0].set_xlabel('$x$')
    axes[0, 0].set_ylabel('$u(x)$')
    axes[0, 0].legend()
    axes[0, 0].set_title('Solution with different $\\lambda$')
    axes[0, 0].grid(True, alpha=0.3)

    # PDE损失
    for i, lam in enumerate(lambda_values):
        axes[0, 1].semilogy(results[lam]['history']['pde'],
                            linewidth=1.5, color=colors[i], label=f'$\\lambda$={lam}', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PDE Loss')
    axes[0, 1].legend()
    axes[0, 1].set_title('PDE loss evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # BC损失
    for i, lam in enumerate(lambda_values):
        axes[1, 0].semilogy(results[lam]['history']['bc'],
                            linewidth=1.5, color=colors[i], label=f'$\\lambda$={lam}', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BC Loss')
    axes[1, 0].legend()
    axes[1, 0].set_title('Boundary loss evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # MSE vs lambda
    mse_values = [results[lam]['mse'] for lam in lambda_values]
    axes[1, 1].semilogx(lambda_values, mse_values, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('$\\lambda$')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('MSE vs penalty parameter')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lambda_effect.png', dpi=150)
    plt.show()

    # 打印汇总表格
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"{'λ':<10} {'Final PDE Loss':<18} {'Final BC Loss':<18} {'MSE':<15}")
    print("-"*60)
    for lam in lambda_values:
        print(f"{lam:<10} {results[lam]['history']['pde'][-1]:<18.2e} "
              f"{results[lam]['history']['bc'][-1]:<18.2e} {results[lam]['mse']:<15.2e}")
