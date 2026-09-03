#!/usr/bin/env python3
"""
第8章：深度学习与自动微分演示

实现一个迷你版的自动微分系统（类似PyTorch的核心机制）
"""
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第8章：深度学习与自动微分演示")
print("="*60)

# ==================== 自动微分引擎 ====================
print("\n" + "="*60)
print("Part 1: 迷你自动微分引擎")
print("="*60)

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
        topo = []
        visited = set()

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


# ==================== 测试自动微分 ====================
print("\n测试1：y = x^2 + 2x + 1，求 dy/dx")
print("-" * 40)

x = Variable(3.0)
y = x * x + x * 2 + 1
y.backward()

print(f"x = {x.data}")
print(f"y = x^2 + 2x + 1 = {y.data}")
print(f"dy/dx (自动微分) = {x.grad}")
print(f"dy/dx (解析: 2x + 2) = {2*x.data + 2}")
assert abs(x.grad - 8.0) < 1e-6, "梯度计算错误！"
print("✓ 验证通过！")

# 测试更复杂的表达式
print("\n测试2：z = sin(x) 的泰勒展开验证")
print("-" * 40)

x2 = Variable(np.pi / 4)
# 用泰勒展开近似 sin(x) ≈ x - x^3/6 + x^5/120
z = x2 - (x2 ** 3) / 6 + (x2 ** 5) / 120
z.backward()

print(f"x = π/4 = {x2.data:.6f}")
print(f"sin(x) (泰勒近似) = {z.data:.6f}")
print(f"sin(x) (真实值) = {np.sin(np.pi/4):.6f}")
print(f"d(sin)/dx (自动微分) = {x2.grad:.6f}")
print(f"cos(x) (真实导数) = {np.cos(np.pi/4):.6f}")

# 测试3：复合函数
print("\n测试3：f(x) = tanh(x^2 + 1)")
print("-" * 40)

x3 = Variable(0.5)
f = (x3 * x3 + 1).tanh()
f.backward()

# 手工计算：f(x) = tanh(x^2 + 1)
# df/dx = sech^2(x^2+1) * 2x = (1 - tanh^2(x^2+1)) * 2x
val = x3.data ** 2 + 1
expected_grad = (1 - np.tanh(val)**2) * 2 * x3.data

print(f"x = {x3.data}")
print(f"f(x) = tanh(x^2 + 1) = {f.data:.6f}")
print(f"df/dx (自动微分) = {x3.grad:.6f}")
print(f"df/dx (解析) = {expected_grad:.6f}")
assert abs(x3.grad - expected_grad) < 1e-6, "梯度计算错误！"
print("✓ 验证通过！")

# 测试4：ReLU激活
print("\n测试4：ReLU激活函数")
print("-" * 40)

x_pos = Variable(2.0)
y_pos = (x_pos * 3 - 1).relu()
y_pos.backward()
print(f"ReLU(3*2-1) = ReLU(5) = {y_pos.data}, 梯度 = {x_pos.grad}")

x_neg = Variable(-1.0)
y_neg = (x_neg * 3 + 2).relu()
y_neg.backward()
print(f"ReLU(3*(-1)+2) = ReLU(-1) = {y_neg.data}, 梯度 = {x_neg.grad}")

# ==================== 简单神经网络 ====================
print("\n" + "="*60)
print("Part 2: 用自动微分训练单个神经元")
print("="*60)

# 训练数据：简单的线性回归
X_train = [1.0, 2.0, 3.0, 4.0]
Y_train = [2.1, 4.0, 5.9, 8.1]  # 近似 y = 2x

# 参数
w = Variable(0.5)  # 权重
b = Variable(0.0)  # 偏置

losses = []
lr = 0.01

print(f"\n目标：学习 y ≈ 2x")
print(f"初始参数：w = {w.data:.4f}, b = {b.data:.4f}")

for epoch in range(100):
    # 清零梯度
    w.grad = 0.0
    b.grad = 0.0

    # 计算总损失
    total_loss = Variable(0.0)
    for x, y in zip(X_train, Y_train):
        pred = w * x + b
        loss = (pred - y) ** 2
        total_loss = total_loss + loss

    total_loss.backward()
    losses.append(total_loss.data)

    # 梯度下降
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data:.4f}, w = {w.data:.4f}, b = {b.data:.4f}")

print(f"\n最终参数：w = {w.data:.4f}, b = {b.data:.4f}")
print(f"期望：w ≈ 2.0, b ≈ 0")

# ==================== 反向传播与拉格朗日乘子的联系 ====================
print("\n" + "="*60)
print("Part 3: 反向传播 = 拉格朗日乘子法")
print("="*60)

print("""
书中核心洞察：
反向传播中的"误差信号" δ_i = -λ_i 就是拉格朗日乘子！

对于神经网络的约束优化形式：
  min L(z_n)
  s.t. z_1 = f_1(x; θ_1)
       z_2 = f_2(z_1; θ_2)
       ...
       z_n = f_n(z_{n-1}; θ_n)

拉格朗日函数：
  L = ℓ(z_n) + Σ λ_i^T (z_i - f_i(z_{i-1}; θ_i))

KKT条件给出伴随方程（反向传播公式）：
  λ_n = -∇ℓ(z_n)          (终端条件)
  λ_i = (∂f_{i+1}/∂z_i)^T λ_{i+1}  (递推公式)

这正是反向传播算法！
""")

# 演示：验证两层网络的梯度
print("数值验证：两层网络的梯度")
print("-" * 40)

# 简化的两层网络
x_input = 1.5
w1 = Variable(0.7)
w2 = Variable(-0.3)
target = 1.0

# 前向传播
z1 = (w1 * x_input).tanh()  # 隐藏层
z2 = w2 * z1                 # 输出层
loss = (z2 - target) ** 2   # 损失

# 反向传播
loss.backward()

print(f"前向传播：")
print(f"  z1 = tanh(w1 * x) = tanh({w1.data:.4f} * {x_input}) = {z1.data:.4f}")
print(f"  z2 = w2 * z1 = {w2.data:.4f} * {z1.data:.4f} = {z2.data:.4f}")
print(f"  loss = (z2 - target)^2 = ({z2.data:.4f} - {target})^2 = {loss.data:.4f}")
print(f"\n反向传播（自动微分）：")
print(f"  dL/dw1 = {w1.grad:.6f}")
print(f"  dL/dw2 = {w2.grad:.6f}")

# 数值梯度验证
eps = 1e-5
w1_test = 0.7
z1_p = np.tanh((w1_test + eps) * x_input)
z1_m = np.tanh((w1_test - eps) * x_input)
z2_p = w2.data * z1_p
z2_m = w2.data * z1_m
loss_p = (z2_p - target) ** 2
loss_m = (z2_m - target) ** 2
numerical_grad_w1 = (loss_p - loss_m) / (2 * eps)

print(f"\n数值梯度验证：")
print(f"  dL/dw1 (数值) = {numerical_grad_w1:.6f}")
print(f"  误差 = {abs(w1.grad - numerical_grad_w1):.2e}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：自动微分计算图示意
ax1 = axes[0, 0]
x_range = np.linspace(-3, 5, 100)
y_range = x_range**2 + 2*x_range + 1
dy_range = 2*x_range + 2

ax1.plot(x_range, y_range, 'b-', linewidth=2, label='$y = x^2 + 2x + 1$')
ax1.plot(x_range, dy_range, 'r--', linewidth=2, label="$y' = 2x + 2$")
ax1.scatter([3], [16], color='blue', s=100, zorder=5)
ax1.scatter([3], [8], color='red', s=100, zorder=5)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# 画切线
x0 = 3
y0 = x0**2 + 2*x0 + 1
slope = 2*x0 + 2
tangent_x = np.linspace(1, 5, 50)
tangent_y = slope * (tangent_x - x0) + y0
ax1.plot(tangent_x, tangent_y, 'g-', linewidth=1.5, alpha=0.7, label=f'x=3处切线 (斜率={slope})')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12, labelpad=10)
ax1.set_title('自动微分: 求dy/dx\n$y = x^2 + 2x + 1$, 验证x=3时dy/dx=8', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：训练损失曲线
ax2 = axes[0, 1]
ax2.plot(losses, 'b-', linewidth=1.5)
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('损失值', fontsize=12, labelpad=10)
ax2.set_title('线性回归训练损失\n(学习 y = 2x)', fontsize=14)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 图3：线性回归拟合结果
ax3 = axes[1, 0]
ax3.scatter(X_train, Y_train, color='blue', s=100, label='训练数据')
x_line = np.linspace(0, 5, 100)
y_line = w.data * x_line + b.data
ax3.plot(x_line, y_line, 'r-', linewidth=2, label=f'学习结果: y = {w.data:.2f}x + {b.data:.2f}')
ax3.plot(x_line, 2*x_line, 'g--', linewidth=1, alpha=0.5, label='真实: y = 2x')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12, labelpad=10)
ax3.set_title('线性回归拟合结果', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：反向传播与拉格朗日乘子的对应关系
ax4 = axes[1, 1]
ax4.axis('off')

# 创建对比表格
table_data = [
    ['神经网络', '最优控制'],
    ['层: z_i+1 = f_i(z_i; θ_i)', '动力学: dx/dt = f(x, u)'],
    ['损失: L = l(z_n)', '终端代价: φ(x(T))'],
    ['伴随变量: λ_i', '协态变量: p(t)'],
    ['λ_n = -∇l', 'p(T) = -∇φ'],
    ['λ_i = (∂f/∂z)^T λ_i+1', 'dp/dt = -∂H/∂x'],
]

ax4.text(0.5, 0.95, '统一视角: 反向传播 = 伴随方法',
         fontsize=14, fontweight='bold', ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.88, '(都是约束优化的KKT条件!)',
         fontsize=11, ha='center', transform=ax4.transAxes, style='italic')

y_pos = 0.75
for row in table_data:
    if row == table_data[0]:
        ax4.text(0.25, y_pos, row[0], fontsize=12, fontweight='bold', ha='center', transform=ax4.transAxes)
        ax4.text(0.75, y_pos, row[1], fontsize=12, fontweight='bold', ha='center', transform=ax4.transAxes)
    else:
        ax4.text(0.25, y_pos, row[0], fontsize=10, ha='center', transform=ax4.transAxes)
        ax4.text(0.75, y_pos, row[1], fontsize=10, ha='center', transform=ax4.transAxes)
    y_pos -= 0.1

# 画分隔线
ax4.axhline(y=0.71, xmin=0.05, xmax=0.95, color='black', linewidth=1)
ax4.axvline(x=0.5, ymin=0.1, ymax=0.95, color='gray', linewidth=0.5, linestyle='--')

ax4.text(0.5, 0.08, '核心洞察: 拉格朗日乘子λ代表约束的"影子价格"',
         fontsize=10, ha='center', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('figs/chap08_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap08_fig1.png")

plt.show()
print("\n第8章代码执行完成！")
