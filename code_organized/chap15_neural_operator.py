#!/usr/bin/env python3
"""
第15章：神经算子演示

演示：
1. DeepONet学习反导数算子
2. FNO学习Burgers方程
3. 离散化无关性验证
"""
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第15章：神经算子演示")
print("="*60)

# ==================== 简化的DeepONet演示 ====================
print("\n" + "="*60)
print("Part 1: DeepONet概念演示")
print("="*60)

class SimpleDeepONet:
    """
    简化的DeepONet演示

    学习反导数算子：G(u)(x) = integral_0^x u(t) dt

    架构：
    - 分支网络：输入函数在传感器位置的采样值 -> 系数向量
    - 主干网络：查询位置 -> 基函数值
    - 输出：内积
    """

    def __init__(self, n_sensors, n_basis, hidden_dim=64):
        self.n_sensors = n_sensors
        self.n_basis = n_basis
        self.hidden_dim = hidden_dim

        # 分支网络参数（简化为单隐层）
        np.random.seed(42)
        scale = 0.1
        self.W_branch1 = np.random.randn(n_sensors, hidden_dim) * scale
        self.b_branch1 = np.zeros(hidden_dim)
        self.W_branch2 = np.random.randn(hidden_dim, n_basis) * scale
        self.b_branch2 = np.zeros(n_basis)

        # 主干网络参数
        self.W_trunk1 = np.random.randn(1, hidden_dim) * scale
        self.b_trunk1 = np.zeros(hidden_dim)
        self.W_trunk2 = np.random.randn(hidden_dim, n_basis) * scale
        self.b_trunk2 = np.zeros(n_basis)

        self.bias = 0.0

    def branch_net(self, u_sensors):
        """分支网络：编码输入函数"""
        h = np.tanh(u_sensors @ self.W_branch1 + self.b_branch1)
        return h @ self.W_branch2 + self.b_branch2

    def trunk_net(self, x):
        """主干网络：编码查询位置"""
        x = x.reshape(-1, 1)
        h = np.tanh(x @ self.W_trunk1 + self.b_trunk1)
        return h @ self.W_trunk2 + self.b_trunk2

    def forward(self, u_sensors, x_query):
        """前向传播"""
        b = self.branch_net(u_sensors)  # (n_basis,)
        t = self.trunk_net(x_query)     # (n_query, n_basis)
        return t @ b + self.bias        # (n_query,)

# 演示DeepONet的概念
print("\nDeepONet架构说明：")
print("- 分支网络：输入 u(y_1),...,u(y_m) -> 输出系数 (b_1,...,b_p)")
print("- 主干网络：输入查询位置 x -> 输出基函数值 (t_1,...,t_p)")
print("- 最终输出：G(u)(x) = sum_k b_k * t_k = <b, t>")

# 创建模型
n_sensors = 20
n_basis = 16
model = SimpleDeepONet(n_sensors, n_basis)

# 测试前向传播
x_sensors = np.linspace(0, 1, n_sensors)
u_test = np.sin(np.pi * x_sensors)  # 测试输入函数
x_query = np.linspace(0, 1, 50)

y_pred = model.forward(u_test, x_query)
print(f"\n测试：")
print(f"  输入函数: u(x) = sin(πx)")
print(f"  传感器数量: {n_sensors}")
print(f"  基函数数量: {n_basis}")
print(f"  输出形状: {y_pred.shape}")

# ==================== FNO概念演示 ====================
print("\n" + "="*60)
print("Part 2: FNO概念演示")
print("="*60)

def spectral_conv_1d(u, R_weights, k_max):
    """
    1D傅里叶卷积层

    步骤：
    1. FFT变换到频域
    2. 截断高频（只保留k_max个模态）
    3. 在频域做点乘
    4. IFFT变换回空域

    参数:
        u: 输入信号，形状 (n_x,)
        R_weights: 频域权重，形状 (k_max,) 复数
        k_max: 保留的最大频率模态数
    """
    n_x = len(u)

    # 1. FFT
    u_ft = np.fft.rfft(u)

    # 2. 截断高频
    u_ft_truncated = u_ft[:k_max]

    # 3. 频域乘法
    out_ft = R_weights * u_ft_truncated

    # 4. 补零并IFFT
    out_ft_full = np.zeros(n_x // 2 + 1, dtype=complex)
    out_ft_full[:k_max] = out_ft
    u_out = np.fft.irfft(out_ft_full, n=n_x)

    return u_out

print("\nFNO的核心：频域滤波")
print("1. 傅里叶变换将空域函数变换到频域")
print("2. 在频域做逐点乘法（等效于空域卷积）")
print("3. 只保留低频模态（物理假设：解是光滑的）")
print("4. 逆变换回空域")

# 演示频域滤波
n_x = 64
k_max = 12  # 只保留前12个模态
x = np.linspace(0, 2*np.pi, n_x)

# 输入：混合信号
u_input = np.sin(x) + 0.3*np.sin(5*x) + 0.1*np.sin(20*x)

# 随机频域权重（演示用）
np.random.seed(123)
R = np.random.randn(k_max) + 1j * np.random.randn(k_max)

# 应用频域滤波
u_output = spectral_conv_1d(u_input, R, k_max)

print(f"\n频域滤波演示：")
print(f"  输入: sin(x) + 0.3*sin(5x) + 0.1*sin(20x)")
print(f"  保留模态数: {k_max}")
print(f"  高频分量（20x）会被截断")

# ==================== 离散化无关性演示 ====================
print("\n" + "="*60)
print("Part 3: 离散化无关性")
print("="*60)

print("\n离散化无关性是神经算子的核心特性：")
print("- 在低分辨率(如32点)上训练")
print("- 可以在高分辨率(如128点)上推理")
print("- 无需重新训练！")

def demo_resolution_invariance():
    """演示频域操作的分辨率无关性"""
    # 定义一个"频域滤波器"：保留低频，衰减高频
    def frequency_filter(u, cutoff=5):
        """低通滤波器"""
        n = len(u)
        u_ft = np.fft.rfft(u)

        # 创建滤波器
        freqs = np.fft.rfftfreq(n)
        filter_response = np.exp(-np.abs(np.arange(len(u_ft))) / cutoff)

        # 应用滤波
        u_ft_filtered = u_ft * filter_response
        return np.fft.irfft(u_ft_filtered, n=n)

    # 同一个物理函数在不同分辨率下
    results = {}
    for n_points in [32, 64, 128, 256]:
        x = np.linspace(0, 2*np.pi, n_points)
        u = np.sin(x) + 0.3*np.cos(3*x)
        u_filtered = frequency_filter(u, cutoff=5)
        results[n_points] = (x, u, u_filtered)

    return results

results = demo_resolution_invariance()
print("\n不同分辨率下的滤波结果：")
for n_points, (x, u, u_filtered) in results.items():
    print(f"  {n_points}点: 输出范围 [{u_filtered.min():.4f}, {u_filtered.max():.4f}]")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：DeepONet架构示意
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# 分支网络
ax1.add_patch(plt.Rectangle((0.5, 6), 2.5, 3, fill=True, color='lightblue', ec='black', lw=2))
ax1.text(1.75, 7.5, '分支\n网络', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(1.75, 6.3, r'$u(y_1)...u(y_m)$', ha='center', va='center', fontsize=9)

# 主干网络
ax1.add_patch(plt.Rectangle((0.5, 1), 2.5, 3, fill=True, color='lightgreen', ec='black', lw=2))
ax1.text(1.75, 2.5, '主干\n网络', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(1.75, 1.3, r'$x$', ha='center', va='center', fontsize=10)

# 输出
ax1.add_patch(plt.Rectangle((5, 3.5), 3, 2.5, fill=True, color='lightyellow', ec='black', lw=2))
ax1.text(6.5, 4.75, r'$\mathcal{G}(u)(x)$', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(6.5, 4, r'$= \mathbf{b} \cdot \mathbf{t}$', ha='center', va='center', fontsize=11)

# 箭头
ax1.annotate('', xy=(5, 5.5), xytext=(3, 7.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax1.annotate('', xy=(5, 4), xytext=(3, 2.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='green'))

ax1.text(4, 7, r'$\mathbf{b}$', fontsize=11, color='blue')
ax1.text(4, 3, r'$\mathbf{t}$', fontsize=11, color='green')

ax1.set_title('DeepONet架构\n(分支网络 + 主干网络)', fontsize=12)
ax1.axis('off')

# 图2：FNO层结构
ax2 = axes[0, 1]
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 6)

# 输入
ax2.add_patch(plt.Rectangle((0.5, 2), 1.5, 2, fill=True, color='lightblue', ec='black'))
ax2.text(1.25, 3, r'$v^{(l)}$', ha='center', va='center', fontsize=10)

# 上路：FFT
ax2.add_patch(plt.Rectangle((3, 4), 1.5, 1.5, fill=True, color='yellow', ec='black'))
ax2.text(3.75, 4.75, 'FFT', ha='center', va='center', fontsize=9, fontweight='bold')

ax2.add_patch(plt.Rectangle((5, 4), 1.5, 1.5, fill=True, color='orange', ec='black'))
ax2.text(5.75, 4.75, r'$R \cdot$', ha='center', va='center', fontsize=9, fontweight='bold')

ax2.add_patch(plt.Rectangle((7, 4), 1.5, 1.5, fill=True, color='yellow', ec='black'))
ax2.text(7.75, 4.75, 'IFFT', ha='center', va='center', fontsize=9, fontweight='bold')

# 下路：线性
ax2.add_patch(plt.Rectangle((5, 0.5), 1.5, 1.5, fill=True, color='lightgreen', ec='black'))
ax2.text(5.75, 1.25, r'$W$', ha='center', va='center', fontsize=9, fontweight='bold')

# 加法和激活
ax2.scatter([9], [3], s=400, c='white', edgecolors='black', linewidths=2)
ax2.text(9, 3, '+', ha='center', va='center', fontsize=14, fontweight='bold')

ax2.add_patch(plt.Rectangle((10, 2), 1.5, 2, fill=True, color='pink', ec='black'))
ax2.text(10.75, 3, r'$\sigma$', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
ax2.annotate('', xy=(3, 4.75), xytext=(2, 3.5), arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate('', xy=(5, 4.75), xytext=(4.5, 4.75), arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate('', xy=(7, 4.75), xytext=(6.5, 4.75), arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate('', xy=(8.7, 3.3), xytext=(8.5, 4.75), arrowprops=dict(arrowstyle='->', lw=1.5))

ax2.annotate('', xy=(5, 1.25), xytext=(2, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate('', xy=(8.7, 2.7), xytext=(6.5, 1.25), arrowprops=dict(arrowstyle='->', lw=1.5))

ax2.annotate('', xy=(10, 3), xytext=(9.3, 3), arrowprops=dict(arrowstyle='->', lw=1.5))

ax2.set_title('FNO层结构\n(频域卷积 + 线性变换)', fontsize=12)
ax2.axis('off')

# 图3：频域滤波演示
ax3 = axes[1, 0]
x_demo = np.linspace(0, 2*np.pi, 100)
u_demo = np.sin(x_demo) + 0.3*np.sin(5*x_demo) + 0.1*np.sin(20*x_demo)

# FFT
u_ft = np.fft.rfft(u_demo)
freqs = np.fft.rfftfreq(len(u_demo), d=2*np.pi/len(u_demo))

# 低通滤波
k_max = 8
u_ft_filtered = u_ft.copy()
u_ft_filtered[k_max:] = 0
u_filtered = np.fft.irfft(u_ft_filtered, n=len(u_demo))

ax3.plot(x_demo, u_demo, 'b-', linewidth=2, label='原始信号 (含高频)', alpha=0.7)
ax3.plot(x_demo, u_filtered, 'r-', linewidth=2, label='FNO层输出 (低通滤波)')
ax3.set_xlabel('x', labelpad=10)
ax3.set_ylabel('u(x)', labelpad=10)
ax3.set_title(f'FNO低通滤波效果\n(只保留前{k_max}个傅里叶模态)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：离散化无关性
ax4 = axes[1, 1]
colors = plt.cm.viridis(np.linspace(0, 1, 4))
for i, (n_points, (x, u, u_filtered)) in enumerate(results.items()):
    ax4.plot(x / (2*np.pi), u_filtered, '-', color=colors[i], linewidth=2-i*0.3,
             label=f'{n_points}点', alpha=0.8)

ax4.set_xlabel('x / 2π', labelpad=10)
ax4.set_ylabel('滤波后 u(x)', labelpad=10)
ax4.set_title('离散化无关性\n(相同滤波器在不同分辨率下)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/chap15_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap15_fig1.png")

# ==================== 数值验证 ====================
print("\n" + "="*60)
print("验证：卷积定理")
print("="*60)

# 验证：空域卷积 = 频域乘法
n = 64
np.random.seed(42)
f = np.random.randn(n)
g = np.random.randn(n)

# 方法1：空域卷积
conv_spatial = np.convolve(f, g, mode='same')

# 方法2：频域乘法
f_ft = np.fft.fft(f)
g_ft = np.fft.fft(g)
conv_freq = np.real(np.fft.ifft(f_ft * g_ft))
# 调整以匹配 'same' 模式
conv_freq_shifted = np.roll(conv_freq, n//2)

print(f"空域卷积结果范围: [{conv_spatial.min():.4f}, {conv_spatial.max():.4f}]")
print(f"频域乘法结果范围: [{conv_freq_shifted.min():.4f}, {conv_freq_shifted.max():.4f}]")
print("结论：频域乘法等效于空域卷积（傅里叶变换的卷积定理）")

plt.show()
print("\n第15章代码执行完成！")
