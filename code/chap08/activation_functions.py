"""
第8章扩展：手写激活函数与计算图
=================================

从仓库根目录运行::

    python3 code/chap08/activation_functions.py

输出（pdf + png）：
- figures/chap08/activation_functions   8 种激活函数及其导数
- figures/chap08/gradient_flow          (a) 三种激活函数的导数分布；(b) 深层网络各层梯度范数（真实前向/反向计算）；
                                     (c) "死亡 ReLU"：真实训练实验中输出恒为 0 的隐藏单元比例
- figures/chap08/computation_graph      单层神经元的前向 / 反向计算图（示意图）
- figures/chap08/gradient_check         解析梯度 vs 中心差分数值梯度及其误差
- figures/chap08/modern_activations     ReLU / GELU / Swish / Mish 及其导数（导数由 torch 自动微分求得）

依赖：numpy、matplotlib、torch（CPU 即可）。
"""

import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

warnings.filterwarnings('ignore')
setup_style()


def _torch():
    """按需导入 torch，并固定单线程（小网络上多线程反而因线程争用变慢）。"""
    import torch
    torch.set_num_threads(1)
    return torch

# 各激活函数在本章图中的固定配色
C = {'sigmoid': COLORS['blue'], 'tanh': COLORS['green'], 'relu': COLORS['red'],
     'leaky': COLORS['orange'], 'gelu': COLORS['blue'], 'swish': COLORS['orange'], 'mish': COLORS['purple']}


# =============================================================================
# 第1部分：激活函数实现
# =============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_func(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(np.minimum(x, 0)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_derivative(x):
    return sigmoid(x)


def swish(x):
    return x * sigmoid(x)


def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    du = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + np.tanh(u)) + 0.5 * x * (1 - np.tanh(u) ** 2) * du


def mish(x):
    return x * np.tanh(softplus(x))


# =============================================================================
# 图 1：激活函数全家福
# =============================================================================
def plot_activation_functions(save_path):
    x = np.linspace(-4, 4, 500)
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.3), sharex=True, sharey=True)
    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative), ('Tanh', tanh_func, tanh_derivative),
        ('ReLU', relu, relu_derivative), ('Leaky ReLU', leaky_relu, leaky_relu_derivative),
        ('ELU', elu, elu_derivative), ('Softplus', softplus, softplus_derivative),
        ('Swish', swish, swish_derivative), ('GELU', gelu, gelu_derivative),
    ]
    for idx, (name, func, deriv) in enumerate(activations):
        ax = axes[idx // 4, idx % 4]
        ax.axhline(0, color=COLORS['gray'], lw=0.5)
        ax.axvline(0, color=COLORS['gray'], lw=0.5)
        ax.plot(x, func(x), color=COLORS['blue'], lw=1.5, label='激活函数 $f(x)$')
        ax.plot(x, deriv(x), color=COLORS['orange'], lw=1.3, ls='--', label="导数 $f'(x)$")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 2.5)
        ax.set_title(name, fontsize=8.5, pad=3)
        ax.tick_params(labelsize=7)
        panel_label(ax, f'({"abcdefgh"[idx]})', x=-0.02, y=1.02)
        if idx // 4 == 1:
            ax.set_xlabel('$x$', fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(w_pad=0.8, h_pad=1.4, rect=(0, 0, 1, 0.93))
    save_figure(fig, save_path)


# =============================================================================
# 图 2：梯度流动（真实计算 + 真实训练实验）
# =============================================================================
def layerwise_gradient_norms(act, depth=20, width=64, init='xavier', seeds=5, batch=256):
    """随机初始化的深层全连接网络：对每一层的激活值 h_l 求 ∂L/∂h_l，返回各层梯度范数（多种子平均）。"""
    torch = _torch()
    out = []
    for s in range(seeds):
        torch.manual_seed(s)
        layers = []
        for _ in range(depth):
            lin = torch.nn.Linear(width, width)
            if init == 'xavier':
                torch.nn.init.xavier_normal_(lin.weight)
            else:
                torch.nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            torch.nn.init.zeros_(lin.bias)
            layers.append(lin)
        h = torch.randn(batch, width).requires_grad_(True)
        hs = [h]
        for lin in layers:
            h = act(lin(h))
            h.retain_grad()
            hs.append(h)
        loss = (h ** 2).sum(1).mean()          # 一个简单的标量损失
        loss.backward()
        out.append([hh.grad.norm(dim=1).mean().item() for hh in hs])
    return np.array(out).mean(0)               # 索引 0 = 输入层，depth = 输出层


def dead_relu_experiment(act_name, lr, epochs=150, width=64, depth=3, seeds=4, n_data=512):
    """真实训练：3 隐藏层 MLP 回归 sin(2x)cos(2y)，Adam 优化器；每个 epoch 统计
    "在整个训练集上预激活 z<=0 的隐藏单元" 的比例（ReLU 下这些单元输出恒为 0 且梯度为 0，即"死亡"）。"""
    torch = _torch()
    g = torch.Generator().manual_seed(1)
    X = torch.rand(n_data, 2, generator=g) * 4 - 2
    Y = (torch.sin(2 * X[:, 0]) * torch.cos(2 * X[:, 1])).unsqueeze(1)
    hist = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.ReLU() if act_name == 'relu' else torch.nn.LeakyReLU(0.1)]
            d = width
        layers.append(torch.nn.Linear(d, 1))
        net = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        h_seed = []
        for ep in range(epochs):
            with torch.no_grad():
                h, dead, tot = X, 0, 0
                for m in net:
                    if isinstance(m, torch.nn.Linear) and m.out_features == width:
                        z = m(h)
                        dead += (z <= 0).all(0).sum().item()
                        tot += width
                        h = z
                    elif not isinstance(m, torch.nn.Linear):
                        h = m(h)
                h_seed.append(dead / tot)
            perm = torch.randperm(n_data)
            for i in range(0, n_data, 64):
                idx = perm[i:i + 64]
                loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        hist.append(h_seed)
    return np.array(hist) * 100


def plot_gradient_flow(save_path):
    torch = _torch()
    x = np.linspace(-4, 4, 500)
    fig, axes = plt.subplots(1, 3, figsize=fig_size(3, 1, aspect=0.95))

    # (a) 导数分布
    ax1 = axes[0]
    ax1.fill_between(x, 0, sigmoid_derivative(x), alpha=0.35, color=C['sigmoid'], label='Sigmoid')
    ax1.fill_between(x, 0, tanh_derivative(x), alpha=0.35, color=C['tanh'], label='Tanh')
    ax1.fill_between(x, 0, relu_derivative(x), alpha=0.35, color=C['relu'], label='ReLU')
    ax1.plot(x, sigmoid_derivative(x), color=C['sigmoid'], lw=1.2)
    ax1.plot(x, tanh_derivative(x), color=C['tanh'], lw=1.2)
    ax1.axhline(0.25, color=C['sigmoid'], ls=':', lw=0.8)
    ax1.annotate("Sigmoid 最大导数仅 0.25", xy=(0.0, 0.25), xytext=(-3.9, 0.62), fontsize=7, color=C['sigmoid'],
                 arrowprops=dict(arrowstyle='->', color=C['sigmoid'], lw=0.8))
    ax1.set_xlabel('输入 $x$')
    ax1.set_ylabel("导数 $f'(x)$")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, 1.6)
    ax1.legend(loc='upper left', ncol=3, fontsize=7, columnspacing=0.6, handlelength=1.0, handletextpad=0.4)
    panel_label(ax1, '(a)')

    # (b) 深层网络各层梯度范数（真实计算）
    ax2 = axes[1]
    depth = 20
    specs = [('Sigmoid（Xavier 初始化）', torch.sigmoid, 'xavier', C['sigmoid'], '-'),
             ('Tanh（Xavier 初始化）', torch.tanh, 'xavier', C['tanh'], '-'),
             ('ReLU（Xavier 初始化）', torch.relu, 'xavier', C['relu'], '-'),
             ('ReLU（He 初始化）', torch.relu, 'he', C['relu'], '--')]
    k = np.arange(depth + 1)
    for name, act, init, col, ls in specs:
        norms = layerwise_gradient_norms(act, depth=depth, init=init)
        ratio = norms[::-1] / norms[-1]          # 距输出 k 层处的梯度范数 / 输出层梯度范数
        ax2.semilogy(k, ratio, color=col, ls=ls, label=name)
        print(f'  梯度范数比（距输出 20 层）{name}: {ratio[-1]:.2e}')
    ax2.semilogy(k, 0.25 ** k, color=COLORS['gray'], ls=':', lw=1.0, label='参考 $0.25^{k}$')
    ax2.set_xlabel('距输出层的层数 $k$')
    ax2.set_ylabel('梯度范数比')
    ax2.set_xlim(0, depth)
    ax2.set_ylim(1e-16, 1e3)
    ax2.legend(loc='lower left', fontsize=6.5, handlelength=1.6)
    panel_label(ax2, '(b)')

    # (c) 死亡 ReLU（真实训练实验）
    ax3 = axes[2]
    runs = [('ReLU，学习率 0.03', 'relu', 0.03, C['relu'], '-'),
            ('Leaky ReLU，学习率 0.03', 'leaky', 0.03, C['leaky'], '--'),
            ('ReLU，学习率 0.001', 'relu', 0.001, C['relu'], ':')]
    for name, act_name, lr, col, ls in runs:
        H = dead_relu_experiment(act_name, lr)
        ep = np.arange(H.shape[1])
        ax3.plot(ep, H.mean(0), color=col, ls=ls, label=name)
        ax3.fill_between(ep, H.min(0), H.max(0), color=col, alpha=0.15, lw=0)
        print(f'  死亡单元比例 {name}: 初始 {H.mean(0)[0]:.1f}% → 结束 {H.mean(0)[-1]:.1f}%')
    ax3.set_xlabel('训练轮数（epoch）')
    ax3.set_ylabel('死亡单元比例 (%)')
    ax3.set_xlim(0, H.shape[1] - 1)
    ax3.set_ylim(0, 110)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.legend(loc='upper left', fontsize=6.8)
    panel_label(ax3, '(c)')

    fig.tight_layout(w_pad=1.2)
    save_figure(fig, save_path)


# =============================================================================
# 图 3：计算图（示意图）
# =============================================================================
def _box(ax, xy, text, w=0.95, h=0.55, fc='#EAF2F8', ec=COLORS['black'], fs=9, ls='-'):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle='round,pad=0.04',
                                facecolor=fc, edgecolor=ec, linewidth=1.0, linestyle=ls))
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fs)


def _circle(ax, xy, text, r=0.28, fc='#E4F0E4', fs=10):
    ax.add_patch(Circle(xy, r, facecolor=fc, edgecolor=COLORS['black'], linewidth=1.0))
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fs)


def _arrow(ax, p0, p1, color, text=None, tpos=0.5, toff=(0, 0.2), fs=7.5, rad=0.0):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3, shrinkA=0, shrinkB=0,
                                connectionstyle=f'arc3,rad={rad}'))
    if text:
        xm = p0[0] + (p1[0] - p0[0]) * tpos + toff[0]
        ym = p0[1] + (p1[1] - p0[1]) * tpos + toff[1]
        ax.text(xm, ym, text, ha='center', va='center', fontsize=fs, color=color)


def plot_computation_graph(save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.6))
    # 节点位置（两幅图共用）
    P = {'x': (0.8, 2.3), 'W': (0.8, 1.3), 'mul': (2.4, 1.8), 'b': (2.4, 0.55), 'add': (3.9, 1.8),
         'z': (5.3, 1.8), 'sig': (6.7, 1.8), 'a': (8.1, 1.8), 'L': (9.4, 1.8)}
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10.1)
        ax.set_ylim(-0.2, 3.1)
        ax.axis('off')
        _box(ax, P['x'], '输入 $x$', fc='#E4F0E4')
        _box(ax, P['W'], '权重 $W$', fc='#FDF1DC')
        _box(ax, P['b'], '偏置 $b$', fc='#FDF1DC')
        _circle(ax, P['mul'], '×')
        _circle(ax, P['add'], '+')
        _box(ax, P['z'], '$z$')
        _circle(ax, P['sig'], '$\\sigma$')
        _box(ax, P['a'], '输出 $a$')
        _box(ax, P['L'], '损失 $L$', fc='white', ls='--', ec=COLORS['gray'])

    blue, red = COLORS['blue'], COLORS['red']
    # ---- (a) 前向传播 ----
    _arrow(ax1, (1.28, 2.3), (2.2, 2.02), blue)
    _arrow(ax1, (1.28, 1.3), (2.2, 1.58), blue)
    _arrow(ax1, (2.68, 1.8), (3.62, 1.8), blue, '$Wx$', toff=(0, 0.22))
    _arrow(ax1, (2.4, 0.83), (3.8, 1.52), blue)
    _arrow(ax1, (4.18, 1.8), (4.82, 1.8), blue)
    _arrow(ax1, (5.78, 1.8), (6.42, 1.8), blue)
    _arrow(ax1, (6.98, 1.8), (7.62, 1.8), blue)
    _arrow(ax1, (8.58, 1.8), (8.92, 1.8), COLORS['gray'])
    ax1.text(5.3, 1.25, '$z = Wx + b$', ha='center', fontsize=8, color=blue)
    ax1.text(8.1, 1.25, '$a = \\sigma(z)$', ha='center', fontsize=8, color=blue)
    ax1.text(9.9, 2.85, '前向：信息从左到右', fontsize=8.5, color=blue, ha='right')
    panel_label(ax1, '(a)', x=0.0, y=0.98)

    # ---- (b) 反向传播 ----
    _arrow(ax2, (8.92, 1.8), (8.58, 1.8), red, '$\\dfrac{\\partial L}{\\partial a}$', toff=(0, 0.42))
    _arrow(ax2, (7.62, 1.8), (6.98, 1.8), red)
    _arrow(ax2, (6.42, 1.8), (5.78, 1.8), red, "$\\times\\,\\sigma'(z)$", toff=(0, 0.3))
    _arrow(ax2, (4.82, 1.8), (4.18, 1.8), red, '$\\dfrac{\\partial L}{\\partial z}$', toff=(0, 0.42))
    _arrow(ax2, (3.62, 1.8), (2.68, 1.8), red, '$\\times\\,1$', toff=(0, 0.25))
    _arrow(ax2, (3.8, 1.52), (2.4, 0.83), red, '$\\dfrac{\\partial L}{\\partial b}=\\dfrac{\\partial L}{\\partial z}$',
           tpos=0.55, toff=(0.9, -0.05))
    _arrow(ax2, (2.2, 2.02), (1.28, 2.3), red, '$\\dfrac{\\partial L}{\\partial x}=W^{T}\\dfrac{\\partial L}{\\partial z}$',
           tpos=0.5, toff=(0.0, 0.62))
    _arrow(ax2, (2.2, 1.58), (1.28, 1.3), red, '$\\dfrac{\\partial L}{\\partial W}=\\dfrac{\\partial L}{\\partial z}\\,x^{T}$',
           tpos=0.5, toff=(-0.8, -0.95))
    ax2.text(6.7, 1.1, "$\\sigma'(z)$ 是“阀门”：\n它很小时梯度流不回去", ha='center', va='top', fontsize=7.5, color=red)
    ax2.text(9.9, 2.85, '反向：梯度从右到左（链式法则）', fontsize=8.5, color=red, ha='right')
    panel_label(ax2, '(b)', x=0.0, y=0.98)

    fig.tight_layout(h_pad=0.5)
    save_figure(fig, save_path)


# =============================================================================
# 图 4：梯度检验
# =============================================================================
def plot_gradient_check(save_path):
    x_test = np.linspace(-3, 3, 301)
    eps = 1e-5
    activations = [('Sigmoid', sigmoid, sigmoid_derivative), ('Tanh', tanh_func, tanh_derivative), ('Swish', swish, swish_derivative)]
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1.0]))
    for idx, (name, func, deriv) in enumerate(activations):
        ax, axe = axes[0, idx], axes[1, idx]
        analytical = deriv(x_test)
        numerical = (func(x_test + eps) - func(x_test - eps)) / (2 * eps)
        error = np.abs(analytical - numerical)
        ax.plot(x_test, analytical, color=COLORS['blue'], lw=1.8, label='解析梯度')
        ax.plot(x_test, numerical, color=COLORS['red'], lw=1.3, ls='--', label='数值梯度（中心差分）')
        ax.set_title(name, fontsize=8.5, pad=3)
        ax.set_ylabel("$f'(x)$" if idx == 0 else '')
        ax.tick_params(labelsize=7.5)
        panel_label(ax, f'({"abc"[idx]})', x=-0.02, y=1.02)
        axe.fill_between(x_test, 1e-14, np.maximum(error, 1e-14), color=COLORS['green'], alpha=0.45, lw=0)
        axe.plot(x_test, np.maximum(error, 1e-14), color=COLORS['green'], lw=0.8)
        axe.set_yscale('log')
        axe.set_ylim(1e-14, 1e-7)
        axe.set_yticks([1e-13, 1e-11, 1e-9])
        axe.set_xlabel('$x$')
        axe.set_ylabel('绝对误差' if idx == 0 else '')
        axe.tick_params(labelsize=7.5)
        axe.text(0.97, 0.9, f'最大 {error.max():.1e}', transform=axe.transAxes, ha='right', va='top', fontsize=7)
        print(f'  梯度检验 {name}: 最大绝对误差 {error.max():.2e}（ε={eps}）')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color=COLORS['green'], alpha=0.45)], labels + ['|解析 − 数值|（对数）'],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(w_pad=1.0, h_pad=0.6, rect=(0, 0, 1, 0.93))
    save_figure(fig, save_path)


# =============================================================================
# 图 5：现代激活函数（导数用 torch 自动微分求得）
# =============================================================================
def plot_modern_activations(save_path):
    torch = _torch()
    import torch.nn.functional as F
    xt = torch.linspace(-4, 4, 801, requires_grad=True)
    x = xt.detach().numpy()
    funcs = [('ReLU', F.relu, COLORS['black'], '-'), ('GELU', F.gelu, C['gelu'], '-'),
             ('Swish (SiLU)', F.silu, C['swish'], '-'), ('Mish', F.mish, C['mish'], '-')]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(2, 1, aspect=0.8))
    for name, fn, col, ls in funcs:
        y = fn(xt)
        (dy,) = torch.autograd.grad(y.sum(), xt)
        ax1.plot(x, y.detach().numpy(), color=col, ls=ls, label=name)
        ax2.plot(x, dy.numpy(), color=col, ls=ls, label=name)
    for ax in (ax1, ax2):
        ax.axhline(0, color=COLORS['gray'], lw=0.5)
        ax.axvline(0, color=COLORS['gray'], lw=0.5)
        ax.set_xlim(-4, 4)
        ax.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_ylim(-1, 4)
    ax1.legend(loc='upper left', fontsize=8)
    ax2.set_ylabel("$f'(x)$（自动微分）")
    ax2.set_ylim(-0.3, 1.3)
    ax2.axhline(1, color=COLORS['gray'], lw=0.5, ls='--')
    ax2.annotate('$x=0$ 附近光滑过渡，\n无导数跳跃', xy=(0.05, 0.55), xytext=(1.6, 0.2), fontsize=7.5,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8))
    ax2.annotate('ReLU 导数在 $x=0$ 跳变', xy=(0, 0.5), xytext=(-3.8, 0.9), fontsize=7.5,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8))
    panel_label(ax1, '(a)')
    panel_label(ax2, '(b)')
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, save_path)


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
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=True, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=True, _children=(self,), _op='relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=True, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def demo_autograd():
    print("=" * 50)
    print("自动微分演示：y = sigmoid(w*x + b)")
    x, w, b = Tensor(2.0, True), Tensor(3.0, True), Tensor(1.0, True)
    z = w * x + b
    y = z.sigmoid()
    y.backward()
    sig_z = 1 / (1 + np.exp(-7))
    dsig = sig_z * (1 - sig_z)
    print(f"dy/dx = {x.grad} (链式法则 σ'(z)·w = {dsig * 3:.6f})")
    print(f"dy/dw = {w.grad} (σ'(z)·x = {dsig * 2:.6f})")
    print(f"dy/db = {b.grad} (σ'(z) = {dsig:.6f})")


# =============================================================================
if __name__ == "__main__":
    out = 'figures/chap09'
    print("[1] 激活函数全家福");    plot_activation_functions(f"{out}/activation_functions")
    print("[2] 梯度流动分析");      plot_gradient_flow(f"{out}/gradient_flow")
    print("[3] 计算图");            plot_computation_graph(f"{out}/computation_graph")
    print("[4] 梯度检验");          plot_gradient_check(f"{out}/gradient_check")
    print("[5] 现代激活函数");      plot_modern_activations(f"{out}/modern_activations")
    print("[6] 自动微分演示");      demo_autograd()
    print("完成。")
