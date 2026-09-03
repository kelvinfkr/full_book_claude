"""
第9章扩展：手写激活函数与计算图
演示如何从零实现激活函数及其导数，并可视化计算图

内容：
1. 各种激活函数的实现和导数
2. 激活函数的数值特性分析
3. 计算图可视化
4. 梯度检验
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 第1部分：激活函数实现
# =============================================================================

def sigmoid(x):
    """Sigmoid激活函数: σ(x) = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Sigmoid导数: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_func(x):
    """Tanh激活函数"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh导数: tanh'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU激活函数: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU导数: 1 if x > 0 else 0"""
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: max(αx, x)"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU导数"""
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    """ELU激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """ELU导数"""
    return np.where(x > 0, 1, alpha * np.exp(x))

def softplus(x):
    """Softplus激活函数: log(1 + exp(x))"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplus_derivative(x):
    """Softplus导数 = sigmoid(x)"""
    return sigmoid(x)

def swish(x):
    """Swish激活函数: x * sigmoid(x)"""
    return x * sigmoid(x)

def swish_derivative(x):
    """Swish导数: sigmoid(x) + x * sigmoid'(x)"""
    s = sigmoid(x)
    return s + x * s * (1 - s)

def gelu(x):
    """GELU激活函数（近似形式）: x * Φ(x)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """GELU导数（数值计算）"""
    eps = 1e-5
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)

# =============================================================================
# 第2部分：可视化函数
# =============================================================================

def plot_activation_functions(save_path):
    """绘制常用激活函数"""
    x = np.linspace(-4, 4, 500)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative, 'blue'),
        ('Tanh', tanh_func, tanh_derivative, 'green'),
        ('ReLU', relu, relu_derivative, 'red'),
        ('Leaky ReLU', leaky_relu, leaky_relu_derivative, 'orange'),
        ('ELU', elu, elu_derivative, 'purple'),
        ('Softplus', softplus, softplus_derivative, 'brown'),
        ('Swish', swish, swish_derivative, 'teal'),
        ('GELU', gelu, gelu_derivative, 'navy'),
    ]

    for idx, (name, func, deriv, color) in enumerate(activations):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]

        y = func(x)
        dy = deriv(x)

        ax.plot(x, y, color=color, linewidth=2, label=f'{name}')
        ax.plot(x, dy, color=color, linewidth=2, linestyle='--', alpha=0.7, label="Derivative")
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 2.5)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_gradient_flow(save_path):
    """绘制梯度流动分析（不同激活函数的梯度传播）"""
    x = np.linspace(-4, 4, 500)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. 梯度饱和区域对比
    ax1 = axes[0]
    ax1.fill_between(x, 0, sigmoid_derivative(x), alpha=0.3, color='blue', label='Sigmoid')
    ax1.fill_between(x, 0, tanh_derivative(x), alpha=0.3, color='green', label='Tanh')
    ax1.fill_between(x, 0, relu_derivative(x), alpha=0.3, color='red', label='ReLU')
    ax1.set_xlabel('Input x', fontsize=11)
    ax1.set_ylabel('Gradient magnitude', fontsize=11)
    ax1.set_title('Gradient Saturation Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, 1.2)

    # 2. 多层梯度衰减
    ax2 = axes[1]
    layers = np.arange(1, 21)

    # Sigmoid: 最大梯度0.25，经过n层后梯度约为0.25^n
    sigmoid_grad = 0.25 ** layers
    # Tanh: 最大梯度1，但典型输入下约0.5
    tanh_grad = 0.5 ** layers
    # ReLU: 假设50%神经元激活
    relu_grad = 0.5 ** layers
    # 理想情况
    ideal_grad = 1.0 ** layers

    ax2.semilogy(layers, sigmoid_grad, 'b-o', markersize=4, label='Sigmoid (worst)')
    ax2.semilogy(layers, tanh_grad, 'g-s', markersize=4, label='Tanh')
    ax2.semilogy(layers, relu_grad, 'r-^', markersize=4, label='ReLU (50% active)')
    ax2.semilogy(layers, ideal_grad, 'k--', label='Ideal (no decay)')
    ax2.set_xlabel('Number of layers', fontsize=11)
    ax2.set_ylabel('Gradient magnitude (log)', fontsize=11)
    ax2.set_title('Vanishing Gradient Problem', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 20)

    # 3. 死亡ReLU问题
    ax3 = axes[2]
    np.random.seed(42)
    # 模拟多个神经元在训练过程中的激活率
    n_neurons = 100
    epochs = np.arange(50)

    # 标准ReLU：某些神经元可能永久死亡
    relu_alive = np.ones(len(epochs))
    death_rate = 0.02
    for i in range(1, len(epochs)):
        relu_alive[i] = relu_alive[i-1] * (1 - death_rate)

    # Leaky ReLU：神经元不会完全死亡
    leaky_alive = np.ones(len(epochs)) * 0.98 + 0.02 * np.random.rand(len(epochs))

    ax3.plot(epochs, relu_alive * 100, 'r-', linewidth=2, label='ReLU')
    ax3.plot(epochs, leaky_alive * 100, 'orange', linewidth=2, linestyle='--', label='Leaky ReLU')
    ax3.fill_between(epochs, 0, relu_alive * 100, alpha=0.2, color='red')
    ax3.set_xlabel('Training epochs', fontsize=11)
    ax3.set_ylabel('Active neurons (%)', fontsize=11)
    ax3.set_title('Dead ReLU Problem', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_computation_graph(save_path):
    """绘制带激活函数的前向/反向传播计算图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：前向传播
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('Forward Pass: z = W·x + b, a = σ(z)', fontsize=13, fontweight='bold', pad=15)

    # 节点位置
    nodes_forward = {
        'x': (1, 6), 'W': (1, 4), 'b': (1, 2),
        '*': (3.5, 5), '+': (5.5, 4),
        'z': (7, 4), 'σ': (8.5, 4),
        'a': (9.5, 4)
    }

    # 绘制节点
    for name, (x, y) in nodes_forward.items():
        if name in ['x', 'W', 'b', 'z', 'a']:
            color = '#E8F4EA' if name in ['x', 'W', 'b'] else '#FFF3CD'
            box = FancyBboxPatch((x-0.4, y-0.35), 0.8, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax1.add_patch(box)
            ax1.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            circle = plt.Circle((x, y), 0.35, facecolor='#D1E7DD', edgecolor='black', linewidth=1.5)
            ax1.add_patch(circle)
            ax1.text(x, y, name, ha='center', va='center', fontsize=16, fontweight='bold')

    # 绘制箭头
    arrows_forward = [
        ('x', '*'), ('W', '*'), ('*', '+'), ('b', '+'),
        ('+', 'z'), ('z', 'σ'), ('σ', 'a')
    ]
    for start, end in arrows_forward:
        x1, y1 = nodes_forward[start]
        x2, y2 = nodes_forward[end]
        # 调整起始和结束位置
        if start in ['x', 'W', 'b', 'z', 'a']:
            x1 += 0.4
        else:
            x1 += 0.35
        if end in ['x', 'W', 'b', 'z', 'a']:
            x2 -= 0.4
        else:
            x2 -= 0.35
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # 添加公式标注
    ax1.text(3.5, 3.2, 'W·x', fontsize=10, style='italic')
    ax1.text(5.5, 2.8, 'W·x + b', fontsize=10, style='italic')
    ax1.text(8.5, 2.8, 'σ(z)', fontsize=10, style='italic')

    # 右图：反向传播
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('Backward Pass: Computing Gradients', fontsize=13, fontweight='bold', pad=15)

    # 反向传播节点
    nodes_backward = {
        '∂L/∂a': (9.5, 6), "σ'": (8, 6), '∂L/∂z': (6.5, 6),
        '∂L/∂W': (3, 7), '∂L/∂b': (3, 5), '∂L/∂x': (3, 3),
        'x^T': (4.5, 6.5), '1': (4.5, 5), 'W^T': (4.5, 3.5)
    }

    # 绘制梯度节点
    for name, (x, y) in nodes_backward.items():
        if name.startswith('∂'):
            box = FancyBboxPatch((x-0.65, y-0.35), 1.3, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor='#F8D7DA', edgecolor='black', linewidth=1.5)
            ax2.add_patch(box)
            ax2.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        elif name == "σ'":
            circle = plt.Circle((x, y), 0.35, facecolor='#D1E7DD', edgecolor='black', linewidth=1.5)
            ax2.add_patch(circle)
            ax2.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax2.text(x, y, name, ha='center', va='center', fontsize=10, style='italic')

    # 反向传播箭头
    ax2.annotate('', xy=(8.35, 6), xytext=(8.85, 6),
                arrowprops=dict(arrowstyle='<-', color='red', lw=2))
    ax2.annotate('', xy=(6.5+0.65, 6), xytext=(8-0.35, 6),
                arrowprops=dict(arrowstyle='<-', color='red', lw=2))

    # 分支箭头
    ax2.annotate('', xy=(3+0.65, 7), xytext=(6.5-0.3, 6.2),
                arrowprops=dict(arrowstyle='<-', color='red', lw=1.5, connectionstyle='arc3,rad=0.2'))
    ax2.annotate('', xy=(3+0.65, 5), xytext=(6.5-0.3, 5.8),
                arrowprops=dict(arrowstyle='<-', color='red', lw=1.5))
    ax2.annotate('', xy=(3+0.65, 3), xytext=(6.5-0.3, 5.6),
                arrowprops=dict(arrowstyle='<-', color='red', lw=1.5, connectionstyle='arc3,rad=-0.2'))

    # 公式标注
    ax2.text(7.25, 5.2, "× σ'(z)", fontsize=9, style='italic', color='darkgreen')
    ax2.text(4.8, 7.2, '× x^T', fontsize=9, style='italic', color='darkgreen')
    ax2.text(4.8, 4.5, '× 1', fontsize=9, style='italic', color='darkgreen')
    ax2.text(4.8, 2.8, '× W^T', fontsize=9, style='italic', color='darkgreen')

    # 添加图例
    ax2.text(1, 1.5, "Key insight: σ'(z) controls gradient flow", fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_gradient_check(save_path):
    """梯度检验：数值梯度 vs 解析梯度"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 测试不同激活函数的梯度正确性
    x_test = np.linspace(-3, 3, 100)
    eps = 1e-5

    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative),
        ('Tanh', tanh_func, tanh_derivative),
        ('Swish', swish, swish_derivative),
    ]

    for idx, (name, func, deriv) in enumerate(activations):
        ax = axes[idx]

        # 解析梯度
        analytical = deriv(x_test)

        # 数值梯度
        numerical = (func(x_test + eps) - func(x_test - eps)) / (2 * eps)

        # 相对误差
        error = np.abs(analytical - numerical) / (np.abs(analytical) + 1e-10)

        ax.plot(x_test, analytical, 'b-', linewidth=2, label='Analytical gradient')
        ax.plot(x_test, numerical, 'r--', linewidth=2, label='Numerical gradient')
        ax.fill_between(x_test, 0, error * 10, alpha=0.3, color='green', label='Error (×10)')

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('Gradient', fontsize=11)
        ax.set_title(f'{name}: Gradient Verification', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 标注最大误差
        max_err = np.max(error)
        ax.text(0.95, 0.95, f'Max error: {max_err:.2e}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_modern_activations(save_path):
    """现代激活函数对比：GELU, Swish, Mish"""
    x = np.linspace(-4, 4, 500)

    def mish(x):
        """Mish激活函数: x * tanh(softplus(x))"""
        return x * np.tanh(softplus(x))

    def mish_derivative(x):
        eps = 1e-5
        return (mish(x + eps) - mish(x - eps)) / (2 * eps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 激活函数对比
    ax1 = axes[0]
    ax1.plot(x, relu(x), 'r-', linewidth=2, label='ReLU')
    ax1.plot(x, gelu(x), 'b-', linewidth=2, label='GELU')
    ax1.plot(x, swish(x), 'g-', linewidth=2, label='Swish')
    ax1.plot(x, mish(x), 'm-', linewidth=2, label='Mish')
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Modern Activation Functions', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1, 4)

    # 导数对比
    ax2 = axes[1]
    ax2.plot(x, relu_derivative(x), 'r-', linewidth=2, label='ReLU')
    ax2.plot(x, gelu_derivative(x), 'b-', linewidth=2, label='GELU')
    ax2.plot(x, swish_derivative(x), 'g-', linewidth=2, label='Swish')
    ax2.plot(x, mish_derivative(x), 'm-', linewidth=2, label='Mish')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.axhline(y=1, color='gray', linewidth=0.5, linestyle='--')
    ax2.axvline(x=0, color='gray', linewidth=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("f'(x)", fontsize=12)
    ax2.set_title('Derivatives of Modern Activations', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.3, 1.3)

    # 标注关键特性
    ax2.annotate('Smooth transition\n(no discontinuity)',
                xy=(0, 0.5), xytext=(2, 0.7),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# 第3部分：简易自动微分实现（带激活函数）
# =============================================================================

class Tensor:
    """简易张量类，支持自动微分"""
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid激活"""
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=True, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """ReLU激活"""
        out = Tensor(np.maximum(0, self.data), requires_grad=True, _children=(self,), _op='relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """Tanh激活"""
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=True, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """反向传播"""
        # 拓扑排序
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 从输出到输入反向传播
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

def demo_autograd():
    """演示自动微分"""
    print("=" * 50)
    print("自动微分演示")
    print("=" * 50)

    # 创建张量
    x = Tensor(2.0, requires_grad=True)
    w = Tensor(3.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)

    # 前向传播: y = sigmoid(w*x + b)
    z = w * x + b  # z = 3*2 + 1 = 7
    y = z.sigmoid()  # y = sigmoid(7) ≈ 0.999

    print(f"x = {x.data}, w = {w.data}, b = {b.data}")
    print(f"z = w*x + b = {z.data}")
    print(f"y = sigmoid(z) = {y.data}")

    # 反向传播
    y.backward()

    print(f"\n梯度:")
    print(f"dy/dx = {x.grad}")
    print(f"dy/dw = {w.grad}")
    print(f"dy/db = {b.grad}")

    # 验证
    print(f"\n验证（链式法则）:")
    sig_z = 1 / (1 + np.exp(-7))
    dsig_dz = sig_z * (1 - sig_z)
    dz_dx = 3.0  # w
    dz_dw = 2.0  # x
    dz_db = 1.0
    print(f"dy/dx = σ'(z) * w = {dsig_dz:.6f} * 3 = {dsig_dz * dz_dx:.6f}")
    print(f"dy/dw = σ'(z) * x = {dsig_dz:.6f} * 2 = {dsig_dz * dz_dw:.6f}")
    print(f"dy/db = σ'(z) * 1 = {dsig_dz:.6f}")

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap09"

    print("=" * 60)
    print("Activation Functions and Computation Graph")
    print("=" * 60)

    # 1. 激活函数图
    print("\n[1] Plotting activation functions...")
    plot_activation_functions(f"{output_dir}/activation_functions.pdf")

    # 2. 梯度流动分析
    print("\n[2] Plotting gradient flow analysis...")
    plot_gradient_flow(f"{output_dir}/gradient_flow.pdf")

    # 3. 计算图
    print("\n[3] Plotting computation graph...")
    plot_computation_graph(f"{output_dir}/computation_graph.pdf")

    # 4. 梯度检验
    print("\n[4] Plotting gradient check...")
    plot_gradient_check(f"{output_dir}/gradient_check.pdf")

    # 5. 现代激活函数
    print("\n[5] Plotting modern activations...")
    plot_modern_activations(f"{output_dir}/modern_activations.pdf")

    # 6. 自动微分演示
    print("\n[6] Auto-differentiation demo...")
    demo_autograd()

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)
