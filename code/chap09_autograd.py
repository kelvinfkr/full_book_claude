#!/usr/bin/env python3
"""
第9章：深度学习与自动微分演示
==============================

实现一个迷你版的自动微分系统（类似 PyTorch 的核心机制），并用它：
1. 验证 y = x² + 2x + 1 在 x = 3 处的导数（= 8）；
2. 训练单个神经元做线性回归 y ≈ 2x；
3. 把一个 N 层残差网络 z_{i+1} = z_i + h f(z_i; θ_i) 看成离散最优控制问题，
   用反向传播得到每一层的伴随量 δ_i = ∂ℓ/∂z_i，并与连续最优控制的协态 p(t)
   （由伴随方程 dp/dt = -(∂f/∂x)^T p 数值积分得到）对照——层数越多，两者越接近。

从仓库根目录运行::

    python3 code/chap09_autograd.py      # 输出 figs_chap09/chap09_fig1.{pdf,png}
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

setup_style()

print("=" * 60)
print("第9章：深度学习与自动微分演示")
print("=" * 60)


# ==================== 自动微分引擎 ====================
class Variable:
    """计算图中的一个节点：既存数据，也存梯度"""

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)  # 简化：只处理标量
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Variable(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    def __neg__(self):
        return self * -1

    def __pow__(self, n):
        out = Variable(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += out.grad * n * (self.data ** (n - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def relu(self):
        out = Variable(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Variable(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

    def backward(self):
        """反向传播：拓扑排序 + 逆序执行 _backward"""
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(topo):
            v._backward()


# ==================== 测试1：y = x^2 + 2x + 1 ====================
print("\n测试1：y = x^2 + 2x + 1，求 dy/dx")
x = Variable(3.0)
y = x * x + x * 2 + 1
y.backward()
slope_autograd = x.grad
print(f"x = {x.data}, y = {y.data}, dy/dx (自动微分) = {slope_autograd}, 解析 2x+2 = {2 * x.data + 2}")
assert abs(slope_autograd - 8.0) < 1e-12

print("\n测试2：f(x) = tanh(x^2 + 1)")
x3 = Variable(0.5)
f = (x3 * x3 + 1).tanh()
f.backward()
expected_grad = (1 - np.tanh(x3.data ** 2 + 1) ** 2) * 2 * x3.data
print(f"df/dx (自动微分) = {x3.grad:.6f}, 解析 = {expected_grad:.6f}")
assert abs(x3.grad - expected_grad) < 1e-12

# ==================== 测试3：单个神经元的线性回归 ====================
print("\nPart 2: 用自动微分训练单个神经元（y ≈ 2x）")
X_train = [1.0, 2.0, 3.0, 4.0]
Y_train = [2.1, 4.0, 5.9, 8.1]
w, b = Variable(0.5), Variable(0.0)
losses, lr = [], 0.01
for epoch in range(100):
    w.grad = b.grad = 0.0
    total_loss = Variable(0.0)
    for xi, yi in zip(X_train, Y_train):
        total_loss = total_loss + (w * xi + b - yi) ** 2
    total_loss.backward()
    losses.append(total_loss.data)
    w.data -= lr * w.grad
    b.data -= lr * b.grad
print(f"最终参数：w = {w.data:.4f}, b = {b.data:.4f}（期望 w≈2, b≈0）")


# ==================== Part 3：反向传播的 δ_i 与最优控制的协态 p(t) ====================
print("\nPart 3: 残差网络的反向传播 = 离散最优控制的伴随方程")
T_END = 2.0
Z0 = np.array([1.0, 0.5])          # 初始状态 z_0 = x(0)
Z_STAR = np.array([-0.5, 1.0])     # 终端目标：ℓ(z_N) = ½‖z_N − z*‖²


def W_of_t(t):
    """随"时间"（层深）缓变的权重 θ(t) = (W(t), b(t))"""
    return np.array([[-0.4, -1.2 - 0.8 * t],
                     [1.5, -0.2]])


def b_of_t(t):
    return np.array([0.3 * np.sin(2 * np.pi * t / T_END), 0.0])


def f_np(t, z):
    """连续动力学 dx/dt = f(x, θ(t)) = tanh(W(t) x + b(t))"""
    return np.tanh(W_of_t(t) @ z + b_of_t(t))


def resnet_adjoint(n_layers):
    """用自制 Variable 引擎搭 N 层残差网络并反向传播，返回各层的 (t_i, z_i, δ_i)。"""
    h = T_END / n_layers
    z = [Variable(Z0[0]), Variable(Z0[1])]
    layers = [z]
    for i in range(n_layers):
        t_i = i * h
        W, bb = W_of_t(t_i), b_of_t(t_i)
        f0 = (z[0] * W[0, 0] + z[1] * W[0, 1] + bb[0]).tanh()
        f1 = (z[0] * W[1, 0] + z[1] * W[1, 1] + bb[1]).tanh()
        z = [z[0] + f0 * h, z[1] + f1 * h]           # z_{i+1} = z_i + h f(z_i; θ_i)（前向 Euler）
        layers.append(z)
    loss = ((z[0] - Z_STAR[0]) ** 2 + (z[1] - Z_STAR[1]) ** 2) * 0.5
    loss.backward()
    t = np.arange(n_layers + 1) * h
    zs = np.array([[v.data for v in lay] for lay in layers])
    deltas = np.array([[v.grad for v in lay] for lay in layers])   # δ_i = ∂ℓ/∂z_i
    return t, zs, deltas, loss.data


# 连续问题：先正向积分 x(t)，再反向积分协态 p(t)
sol_x = solve_ivp(f_np, (0, T_END), Z0, dense_output=True, rtol=1e-10, atol=1e-12)
xT = sol_x.sol(T_END)


def adjoint_rhs(t, p):
    """dp/dt = -(∂f/∂x)^T p，其中 ∂f/∂x = diag(1 - tanh²(Wx+b)) W"""
    xt = sol_x.sol(t)
    u = W_of_t(t) @ xt + b_of_t(t)
    J = (1 - np.tanh(u) ** 2)[:, None] * W_of_t(t)
    return -J.T @ p


sol_p = solve_ivp(adjoint_rhs, (T_END, 0), xT - Z_STAR, dense_output=True, rtol=1e-10, atol=1e-12)
t_fine = np.linspace(0, T_END, 400)
p_fine = sol_p.sol(t_fine)

for N in (5, 10, 20, 40, 80, 160):
    t_i, _, d_i, _ = resnet_adjoint(N)
    err = np.max(np.abs(d_i - sol_p.sol(t_i).T))
    print(f"  N={N:4d} 层 (h={T_END / N:.4f})：max_i |δ_i − p(t_i)| = {err:.3e}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2, aspect=0.7))
ax1, ax2, ax3, ax4 = axes.ravel()

# (a) 自动微分验证
x_range = np.linspace(-3, 5, 200)
ax1.plot(x_range, x_range ** 2 + 2 * x_range + 1, color=COLORS['blue'], label='$y = x^2 + 2x + 1$')
ax1.plot(x_range, 2 * x_range + 2, color=COLORS['orange'], ls='--', label="解析导数 $y' = 2x + 2$")
x0, y0 = 3.0, 16.0
tx = np.linspace(1, 5, 50)
ax1.plot(tx, slope_autograd * (tx - x0) + y0, color=COLORS['red'], lw=1.2,
         label=f'$x=3$ 处切线（自动微分斜率 {slope_autograd:.0f}）')
ax1.plot([x0], [y0], 'o', color=COLORS['red'], ms=5)
ax1.plot([x0], [slope_autograd], 'o', color=COLORS['orange'], ms=5)
ax1.axhline(0, color=COLORS['gray'], lw=0.6)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_ylim(-6, 38)
ax1.legend(loc='upper left', fontsize=7.5)
panel_label(ax1, '(a)')

# (b) 训练损失
ax2.semilogy(losses, color=COLORS['blue'])
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('损失 $\\sum_i (w x_i + b - y_i)^2$')
ax2.set_xlim(0, 100)
ax2.text(0.97, 0.95, f'学习率 {lr}，100 步后\n$w={w.data:.3f}$, $b={b.data:.3f}$', transform=ax2.transAxes,
         ha='right', va='top', fontsize=8)
panel_label(ax2, '(b)')

# (c) 拟合结果
x_line = np.linspace(0, 5, 100)
ax3.plot(X_train, Y_train, 'o', color=COLORS['black'], ms=5, label='训练数据')
ax3.plot(x_line, w.data * x_line + b.data, color=COLORS['blue'], label=f'学习结果 $y = {w.data:.2f}x + {b.data:.2f}$')
ax3.plot(x_line, 2 * x_line, color=COLORS['gray'], ls='--', lw=1.0, label='真实关系 $y = 2x$')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$y$')
ax3.legend(loc='upper left', fontsize=7.5)
panel_label(ax3, '(c)')

# (d) 伴随量 δ_i vs 协态 p(t)
ax4.plot(t_fine, p_fine[0], color=COLORS['black'], lw=1.8, zorder=5, label='协态 $p_1(t)$')
ax4.plot(t_fine, p_fine[1], color=COLORS['black'], lw=1.8, ls='--', zorder=5, label='协态 $p_2(t)$')
for N, col, mk, ms, mfc in ((10, COLORS['orange'], 's', 4.5, 'none'), (40, COLORS['blue'], 'o', 3.0, COLORS['blue'])):
    t_i, _, d_i, _ = resnet_adjoint(N)
    ax4.plot(t_i, d_i[:, 0], mk, color=col, ms=ms, mfc=mfc, zorder=3, label=f'$\\delta_i$（$N={N}$ 层）')
    ax4.plot(t_i, d_i[:, 1], mk, color=col, ms=ms, mfc=mfc, zorder=3)
ax4.set_xlabel('时间 $t = i\\,h$（层深度，$h = T/N$）')
ax4.set_ylabel('伴随量 $\\delta_i$ / 协态 $p(t)$')
ax4.set_xlim(0, T_END)
ax4.set_ylim(-1.0, 0.9)
ax4.legend(loc='upper left', fontsize=6.8, handlelength=1.8)
panel_label(ax4, '(d)')
# 插图：max_i |δ_i - p(t_i)| 随 h 的收敛
Ns = np.array([5, 10, 20, 40, 80, 160])
hs = T_END / Ns
errs = np.array([np.max(np.abs(resnet_adjoint(N)[2] - sol_p.sol(np.arange(N + 1) * T_END / N).T)) for N in Ns])
slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
ins = ax4.inset_axes([0.66, 0.64, 0.32, 0.32])
ins.loglog(hs, errs, 'o-', color=COLORS['red'], ms=2.5, lw=1.0)
ins.set_xticks([1e-2, 1e-1, 1]); ins.set_yticks([1e-2, 1e-1])
ins.xaxis.set_minor_locator(plt.NullLocator()); ins.yaxis.set_minor_locator(plt.NullLocator())
ins.grid(False)
ins.set_xlabel('$h$', fontsize=6.5, labelpad=0)
ins.set_ylabel('$\\max_i|\\delta_i-p(t_i)|$', fontsize=6, labelpad=2)
ins.tick_params(labelsize=5.5, pad=1)
ins.text(0.95, 0.08, f'斜率 {slope:.2f}', transform=ins.transAxes, ha='right', va='bottom', fontsize=6.5, color=COLORS['red'])

fig.tight_layout(w_pad=2.0, h_pad=1.6)
save_figure(fig, 'figs_chap09/chap09_fig1')
print("\n第9章代码执行完成！")
