"""
自适应权重PINN

使用拉格朗日对偶方法自动调整边界条件的权重
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

class AdaptiveWeightPINN:
    def __init__(self, model, lr_model=1e-3, lr_lambda=1e-2, rho=1.0):
        """
        自适应权重PINN

        使用拉格朗日对偶方法自动调整边界条件的权重
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr_model)

        # 拉格朗日乘子（可学习）
        self.lambda_bc = 1.0
        self.lr_lambda = lr_lambda
        self.rho = rho

    def train_step(self, x_interior, x_boundary):
        # 计算损失
        residual = compute_pde_residual(self.model, x_interior)
        loss_pde = torch.mean(residual ** 2)

        u_boundary = self.model(x_boundary)
        loss_bc = torch.mean(u_boundary ** 2)

        # 增广拉格朗日
        total_loss = loss_pde + self.lambda_bc * loss_bc + self.rho / 2 * loss_bc ** 2

        # 更新网络参数（原始更新）
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 更新拉格朗日乘子（对偶更新）
        with torch.no_grad():
            self.lambda_bc = max(0.0, self.lambda_bc + self.lr_lambda * loss_bc.item())

        return loss_pde.item(), loss_bc.item(), self.lambda_bc

    def train(self, num_epochs, N_interior=100):
        x_boundary = torch.tensor([[0.0], [1.0]])
        history = {'pde': [], 'bc': [], 'lambda': []}

        for epoch in range(num_epochs):
            x_interior = torch.rand(N_interior, 1)
            loss_pde, loss_bc, lam = self.train_step(x_interior, x_boundary)

            history['pde'].append(loss_pde)
            history['bc'].append(loss_bc)
            history['lambda'].append(lam)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}: PDE={loss_pde:.2e}, BC={loss_bc:.2e}, λ={lam:.2f}")

        return history

def compute_loss_fixed(model, x_interior, x_boundary, lambda_bc=100.0):
    residual = compute_pde_residual(model, x_interior)
    loss_pde = torch.mean(residual ** 2)
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary ** 2)
    total_loss = loss_pde + lambda_bc * loss_bc
    return total_loss, loss_pde, loss_bc

if __name__ == "__main__":
    # 训练自适应权重PINN
    print("Training Adaptive Weight PINN...")
    torch.manual_seed(42)
    model_adaptive = PINN([1, 50, 50, 50, 1])
    trainer = AdaptiveWeightPINN(model_adaptive, lr_model=1e-3, lr_lambda=0.1, rho=10.0)
    adaptive_history = trainer.train(num_epochs=10000)

    # 固定权重训练作为对比
    print("\nTraining Fixed Weight PINN (λ=100)...")
    torch.manual_seed(42)
    model_fixed = PINN([1, 50, 50, 50, 1])
    optimizer_fixed = torch.optim.Adam(model_fixed.parameters(), lr=1e-3)
    x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    fixed_history = {'pde': [], 'bc': []}

    for epoch in range(10000):
        x_interior = torch.rand(100, 1)
        loss, loss_pde, loss_bc = compute_loss_fixed(model_fixed, x_interior, x_boundary, lambda_bc=100.0)

        optimizer_fixed.zero_grad()
        loss.backward()
        optimizer_fixed.step()

        fixed_history['pde'].append(loss_pde.item())
        fixed_history['bc'].append(loss_bc.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1}: PDE={loss_pde.item():.2e}, BC={loss_bc.item():.2e}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # lambda演化
    axes[0, 0].plot(adaptive_history['lambda'], linewidth=1.5, color='purple')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('$\\lambda$')
    axes[0, 0].set_title('Adaptive weight $\\lambda$ evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # PDE损失对比
    axes[0, 1].semilogy(fixed_history['pde'], label='Fixed weight', linewidth=1.5, alpha=0.7)
    axes[0, 1].semilogy(adaptive_history['pde'], label='Adaptive weight', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PDE Loss')
    axes[0, 1].legend()
    axes[0, 1].set_title('PDE loss comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # BC损失对比
    axes[1, 0].semilogy(fixed_history['bc'], label='Fixed weight', linewidth=1.5, alpha=0.7)
    axes[1, 0].semilogy(adaptive_history['bc'], label='Adaptive weight', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BC Loss')
    axes[1, 0].legend()
    axes[1, 0].set_title('Boundary loss comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # 解的对比
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u_exact = np.sin(np.pi * x_test.numpy()) / (np.pi ** 2)
    with torch.no_grad():
        u_adaptive = model_adaptive(x_test).numpy()
        u_fixed = model_fixed(x_test).numpy()

    axes[1, 1].plot(x_test.numpy(), u_exact, 'b-', linewidth=2, label='Exact')
    axes[1, 1].plot(x_test.numpy(), u_fixed, 'r--', linewidth=2, label='Fixed weight')
    axes[1, 1].plot(x_test.numpy(), u_adaptive, 'g-.', linewidth=2, label='Adaptive weight')
    axes[1, 1].set_xlabel('$x$')
    axes[1, 1].set_ylabel('$u(x)$')
    axes[1, 1].legend()
    axes[1, 1].set_title('Solution comparison')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_weight.png', dpi=150)
    plt.show()

    print(f"\nFinal lambda: {adaptive_history['lambda'][-1]:.2f}")
    print(f"MSE - Fixed: {np.mean((u_fixed - u_exact)**2):.2e}")
    print(f"MSE - Adaptive: {np.mean((u_adaptive - u_exact)**2):.2e}")
