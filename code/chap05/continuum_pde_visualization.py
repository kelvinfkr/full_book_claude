"""
第5章扩展：连续介质力学PDE可视化
展示波动方程、热传导方程、Poisson方程的输入输出和解的形态
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 波动方程可视化
# =============================================================================

def plot_wave_equation(save_path):
    """展示波动方程的初值问题和解的演化"""
    fig = plt.figure(figsize=(14, 10))

    # 参数
    L = 1.0  # 弦长
    c = 1.0  # 波速
    nx = 100
    x = np.linspace(0, L, nx)

    # 初始条件：高斯波包
    x0 = 0.3
    sigma = 0.08
    u0 = np.exp(-(x - x0)**2 / (2 * sigma**2))

    # d'Alembert解：u(x,t) = [f(x-ct) + f(x+ct)]/2
    def wave_solution(x, t, x0=0.3, sigma=0.08, c=1.0):
        # 周期边界
        def f(xi):
            xi = np.mod(xi, 2*L)
            xi = np.where(xi > L, 2*L - xi, xi)
            return np.exp(-(xi - x0)**2 / (2 * sigma**2))
        return 0.5 * (f(x - c*t) + f(x + c*t))

    # 左上：问题设置
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.text(0.5, 0.95, '波动方程问题设置', fontsize=14, fontweight='bold',
            ha='center', va='top', transform=ax1.transAxes)

    # 画弦
    ax1.plot([0, 1], [0.5, 0.5], 'k-', linewidth=3)
    ax1.plot([0, 0], [0.45, 0.55], 'k-', linewidth=5)  # 左端固定
    ax1.plot([1, 1], [0.45, 0.55], 'k-', linewidth=5)  # 右端固定

    # 初始位移
    y_init = 0.5 + 0.15 * u0
    ax1.fill_between(x, 0.5, y_init, alpha=0.3, color='blue')
    ax1.plot(x, y_init, 'b-', linewidth=2, label='初始位移 u(x,0)')

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(0.2, 0.9)
    ax1.set_xlabel('位置 x', fontsize=11)
    ax1.set_ylabel('位移 u', fontsize=11)
    ax1.legend(loc='upper right')

    # 输入输出框
    ax1.text(0.02, 0.15, '输入：\n• 弦长 L = 1 m\n• 波速 c = 1 m/s\n• 初始位移 u(x,0)\n• 初始速度 ∂u/∂t|₀ = 0',
            fontsize=9, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.text(0.55, 0.15, '输出：\n• 位移场 u(x,t)\n• 在任意时刻 t\n• 任意位置 x 的\n  弦的横向位移',
            fontsize=9, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax1.set_title('问题：给定初始波形，求任意时刻的波形', fontsize=11)

    # 右上：时空图
    ax2 = fig.add_subplot(2, 2, 2)
    t_vals = np.linspace(0, 1.5, 100)
    X, T = np.meshgrid(x, t_vals)
    U = np.zeros_like(X)
    for i, t in enumerate(t_vals):
        U[i, :] = wave_solution(x, t)

    im = ax2.pcolormesh(X, T, U, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, ax=ax2, label='位移 u(x,t)')
    ax2.set_xlabel('位置 x', fontsize=11)
    ax2.set_ylabel('时间 t', fontsize=11)
    ax2.set_title('解的时空演化图（颜色=位移）', fontsize=11)

    # 标注波的传播
    ax2.annotate('', xy=(0.6, 0.3), xytext=(0.3, 0.0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax2.annotate('', xy=(0.0, 0.3), xytext=(0.3, 0.0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax2.text(0.45, 0.15, '右行波', fontsize=9, color='white')
    ax2.text(0.1, 0.15, '左行波', fontsize=9, color='white')

    # 左下：不同时刻的波形
    ax3 = fig.add_subplot(2, 2, 3)
    times = [0, 0.15, 0.30, 0.45, 0.60]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    for t, color in zip(times, colors):
        u = wave_solution(x, t)
        ax3.plot(x, u, '-', color=color, linewidth=2, label=f't = {t:.2f}s')

    ax3.set_xlabel('位置 x', fontsize=11)
    ax3.set_ylabel('位移 u(x,t)', fontsize=11)
    ax3.set_title('不同时刻的波形快照', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # 右下：PDE和解的形式
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    text = """
    波动方程（一维弦振动）

    PDE:   ∂²u/∂t² = c² ∂²u/∂x²

    边界条件: u(0,t) = u(L,t) = 0  (两端固定)

    初始条件: u(x,0) = f(x)      (初始形状)
              ∂u/∂t|ₜ₌₀ = g(x)   (初始速度)

    d'Alembert 解:
        u(x,t) = ½[f(x-ct) + f(x+ct)]
               + (1/2c)∫ᵡ⁺ᶜᵗ_{x-ct} g(s)ds

    物理意义:
    • 初始扰动分裂成左右两个行波
    • 波速 c = √(T/ρ)，张力越大、密度越小，波速越快
    • 能量守恒：总能量不随时间变化
    """
    ax4.text(0.05, 0.95, text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle('波动方程：弦的振动', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 2. 热传导方程可视化
# =============================================================================

def plot_heat_equation(save_path):
    """展示热传导方程的初值问题和解的演化"""
    fig = plt.figure(figsize=(14, 10))

    # 参数
    L = 1.0
    alpha = 0.01  # 热扩散系数
    nx = 100
    x = np.linspace(0, L, nx)

    # 初始温度分布
    T0 = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

    # 解析解：傅里叶级数
    def heat_solution(x, t, alpha=0.01, L=1.0):
        # 对于边界条件 T(0)=T(L)=0
        # 初始条件 T(x,0) = sin(πx) + 0.5*sin(3πx)
        n1, n3 = 1, 3
        T = (np.sin(n1 * np.pi * x / L) * np.exp(-(n1 * np.pi / L)**2 * alpha * t) +
             0.5 * np.sin(n3 * np.pi * x / L) * np.exp(-(n3 * np.pi / L)**2 * alpha * t))
        return T

    # 左上：问题设置
    ax1 = fig.add_subplot(2, 2, 1)

    # 画金属棒
    ax1.add_patch(plt.Rectangle((0, 0.4), 1, 0.2, facecolor='gray', edgecolor='black', linewidth=2))
    ax1.text(0.5, 0.5, '金属棒', ha='center', va='center', fontsize=11, color='white')

    # 温度分布用颜色表示
    for i in range(50):
        xi = i / 50
        color = plt.cm.hot(T0[int(xi * (nx-1))] / 1.5 + 0.2)
        ax1.add_patch(plt.Rectangle((xi, 0.65), 0.02, 0.1, facecolor=color, edgecolor='none'))

    ax1.plot(x, 0.65 + 0.3 * T0 / max(T0), 'r-', linewidth=2, label='初始温度分布')

    # 边界条件
    ax1.annotate('T = 0°C\n(冰水)', xy=(0, 0.5), xytext=(-0.15, 0.5),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax1.annotate('T = 0°C\n(冰水)', xy=(1, 0.5), xytext=(1.15, 0.5),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax1.set_xlim(-0.25, 1.25)
    ax1.set_ylim(0.2, 1.1)
    ax1.set_xlabel('位置 x', fontsize=11)
    ax1.axis('off')
    ax1.set_title('问题：给定初始温度分布，求任意时刻的温度', fontsize=11)

    # 输入输出框
    ax1.text(0.02, 0.05, '输入：\n• 棒长 L = 1 m\n• 热扩散系数 α\n• 初始温度 T(x,0)\n• 边界温度 = 0',
            fontsize=9, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.text(0.65, 0.05, '输出：\n• 温度场 T(x,t)\n• 热量随时间\n  从高温区向\n  低温区扩散',
            fontsize=9, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 右上：时空图
    ax2 = fig.add_subplot(2, 2, 2)
    t_vals = np.linspace(0, 50, 100)
    X, T_mesh = np.meshgrid(x, t_vals)
    U = np.zeros_like(X)
    for i, t in enumerate(t_vals):
        U[i, :] = heat_solution(x, t)

    im = ax2.pcolormesh(X, T_mesh, U, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax2, label='温度 T(x,t)')
    ax2.set_xlabel('位置 x', fontsize=11)
    ax2.set_ylabel('时间 t', fontsize=11)
    ax2.set_title('温度的时空演化（颜色=温度）', fontsize=11)

    # 左下：不同时刻的温度分布
    ax3 = fig.add_subplot(2, 2, 3)
    times = [0, 5, 15, 30, 50]
    colors = plt.cm.cool(np.linspace(0, 0.8, len(times)))

    for t, color in zip(times, colors):
        T = heat_solution(x, t)
        ax3.plot(x, T, '-', color=color, linewidth=2, label=f't = {t:.0f}s')

    ax3.set_xlabel('位置 x', fontsize=11)
    ax3.set_ylabel('温度 T(x,t)', fontsize=11)
    ax3.set_title('不同时刻的温度分布', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # 右下：PDE和物理解释
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    text = """
    热传导方程（一维热扩散）

    PDE:   ∂T/∂t = α ∂²T/∂x²

    边界条件: T(0,t) = T(L,t) = 0

    初始条件: T(x,0) = f(x)

    傅里叶级数解:
        T(x,t) = Σ Bₙ sin(nπx/L) exp(-αn²π²t/L²)

    关键特征:
    • 扩散性: 温度从高温向低温扩散
    • 平滑性: 初始的尖锐分布会被"抹平"
    • 衰减性: 高频分量衰减更快
    • 不可逆: 热扩散过程增加熵

    与波动方程对比:
    • 波动: 信息传播，保持形状
    • 热传导: 能量扩散，形状消散
    """
    ax4.text(0.05, 0.95, text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle('热传导方程：温度扩散', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 3. Poisson方程可视化
# =============================================================================

def plot_poisson_equation(save_path):
    """展示Poisson方程（静电场、稳态热传导）"""
    fig = plt.figure(figsize=(14, 10))

    # 网格
    n = 50
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # 源项：中心有一个正电荷
    sigma = 0.1
    rho = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))

    # 简化求解：用迭代法近似
    u = np.zeros((n, n))
    h = 1.0 / (n - 1)
    for _ in range(1000):
        u_old = u.copy()
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                                  u_old[i, j+1] + u_old[i, j-1] + h**2 * rho[i, j])

    # 左上：源项（电荷分布）
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.pcolormesh(X, Y, rho, shading='auto', cmap='Reds')
    plt.colorbar(im1, ax=ax1, label='电荷密度 ρ(x,y)')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('输入：源项（电荷分布）', fontsize=12)
    ax1.set_aspect('equal')

    # 右上：解（电势分布）
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.pcolormesh(X, Y, u, shading='auto', cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='电势 φ(x,y)')
    ax2.contour(X, Y, u, levels=10, colors='white', linewidths=0.5)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('输出：解（电势分布）', fontsize=12)
    ax2.set_aspect('equal')

    # 左下：电场线
    ax3 = fig.add_subplot(2, 2, 3)
    # 计算电场 E = -∇φ
    Ey, Ex = np.gradient(-u, h)

    ax3.streamplot(X, Y, Ex, Ey, color=np.sqrt(Ex**2 + Ey**2),
                   cmap='plasma', density=1.5, linewidth=1)
    ax3.plot(0.5, 0.5, 'r+', markersize=15, markeredgewidth=3)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title('电场线 E = -∇φ', fontsize=12)
    ax3.set_aspect('equal')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # 右下：方程和物理解释
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    text = """
    Poisson 方程（静电场/稳态热传导）

    PDE:   -∇²φ = ρ/ε₀   或   -∇²T = Q/k

    边界条件: φ|∂Ω = 0  (接地边界)

    物理意义:
    • 静电学: φ是电势，ρ是电荷密度
    • 热传导: T是温度，Q是热源

    输入 → 输出:
    • 给定源分布 ρ(x,y)
    • 给定边界条件
    • 求解场分布 φ(x,y)

    关键性质:
    • 椭圆型PDE: 解是光滑的
    • 边界影响全局: 改变边界会改变整个区域的解
    • 最大值原理: 内部极值只能在边界取到
    • 叠加原理: 多个源的效果可以叠加

    数值方法:
    • 有限差分法
    • 有限元法
    • 迭代法（Jacobi, Gauss-Seidel, SOR）
    """
    ax4.text(0.02, 0.98, text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle('Poisson方程：静电场/稳态热传导', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 4. 从离散到连续的过渡
# =============================================================================

def plot_discrete_to_continuum(save_path):
    """展示从离散弹簧系统到连续弦的过渡"""
    fig = plt.figure(figsize=(14, 8))

    # 左上：少量质点（N=5）
    ax1 = fig.add_subplot(2, 2, 1)
    N = 5
    x_discrete = np.linspace(0, 1, N+2)
    u_discrete = 0.3 * np.sin(np.pi * x_discrete)

    # 画弹簧
    for i in range(N+1):
        x_spring = np.linspace(x_discrete[i], x_discrete[i+1], 20)
        y_spring = np.zeros_like(x_spring)
        # 弹簧形状
        for j in range(len(x_spring)):
            y_spring[j] = 0.02 * np.sin(10 * np.pi * (x_spring[j] - x_discrete[i]) / (x_discrete[i+1] - x_discrete[i]))
        y_base = u_discrete[i] + (u_discrete[i+1] - u_discrete[i]) * np.linspace(0, 1, 20)
        ax1.plot(x_spring, y_spring + y_base, 'b-', linewidth=1)

    # 画质点
    ax1.scatter(x_discrete, u_discrete, s=100, c='red', zorder=5)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.5)
    ax1.set_xlabel('位置 x', fontsize=11)
    ax1.set_ylabel('位移 u', fontsize=11)
    ax1.set_title(f'离散系统: N = {N} 个质点', fontsize=12)

    # 右上：更多质点（N=20）
    ax2 = fig.add_subplot(2, 2, 2)
    N = 20
    x_discrete = np.linspace(0, 1, N+2)
    u_discrete = 0.3 * np.sin(np.pi * x_discrete)

    ax2.scatter(x_discrete, u_discrete, s=30, c='red', zorder=5)
    ax2.plot(x_discrete, u_discrete, 'b-', linewidth=0.5, alpha=0.5)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 0.5)
    ax2.set_xlabel('位置 x', fontsize=11)
    ax2.set_ylabel('位移 u', fontsize=11)
    ax2.set_title(f'离散系统: N = {N} 个质点', fontsize=12)

    # 左下：连续极限
    ax3 = fig.add_subplot(2, 2, 3)
    x_cont = np.linspace(0, 1, 200)
    u_cont = 0.3 * np.sin(np.pi * x_cont)

    ax3.fill_between(x_cont, 0, u_cont, alpha=0.3, color='blue')
    ax3.plot(x_cont, u_cont, 'b-', linewidth=2)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 0.5)
    ax3.set_xlabel('位置 x', fontsize=11)
    ax3.set_ylabel('位移 u(x)', fontsize=11)
    ax3.set_title('连续极限: N → ∞ (连续弦)', fontsize=12)

    # 右下：数学对应关系
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    text = """
    从离散到连续的数学对应

    ┌─────────────────────┬───────────────────────────┐
    │      离散系统       │        连续介质           │
    ├─────────────────────┼───────────────────────────┤
    │  质点位移 qᵢ(t)     │   位移场 u(x,t)           │
    │  弹簧刚度 k         │   张力 T / 线密度 ρ       │
    │  求和 Σᵢ           │   积分 ∫dx                │
    │  差分 qᵢ₊₁ - qᵢ    │   导数 ∂u/∂x              │
    │  拉格朗日量 L       │   拉格朗日密度 ℒ          │
    └─────────────────────┴───────────────────────────┘

    离散系统的运动方程:
        m q̈ᵢ = k(qᵢ₊₁ - 2qᵢ + qᵢ₋₁)

              ↓  N → ∞, Δx → 0

    连续系统的PDE:
        ρ ∂²u/∂t² = T ∂²u/∂x²

    关键洞察:
    • 有限自由度 → 无限自由度
    • ODE系统 → PDE
    • 矩阵运算 → 微分算子
    """
    ax4.text(0.02, 0.98, text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle('从离散到连续：场论的起源', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 5. 三类PDE对比
# =============================================================================

def plot_pde_comparison(save_path):
    """对比双曲型、抛物型、椭圆型PDE"""
    fig = plt.figure(figsize=(14, 5))

    # 双曲型：波动方程
    ax1 = fig.add_subplot(1, 3, 1)
    x = np.linspace(0, 1, 100)
    t_vals = [0, 0.2, 0.4]
    for i, t in enumerate(t_vals):
        u = 0.5 * (np.exp(-((x - 0.3 - t) % 1 - 0.5)**2 / 0.01) +
                   np.exp(-((x - 0.3 + t) % 1 - 0.5)**2 / 0.01))
        ax1.plot(x, u + i*0.5, '-', linewidth=2, label=f't = {t}')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('u (偏移显示)', fontsize=11)
    ax1.set_title('双曲型: 波动方程\n∂²u/∂t² = c² ∂²u/∂x²', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.text(0.5, -0.15, '特征: 波传播,\n信息以有限速度传递',
            fontsize=9, ha='center', transform=ax1.transAxes)

    # 抛物型：热方程
    ax2 = fig.add_subplot(1, 3, 2)
    for i, t in enumerate([0, 0.1, 0.5]):
        u = np.exp(-((x - 0.5)**2) / (0.01 + 0.2*t))
        u = u / max(u)  # 归一化
        ax2.plot(x, u + i*0.5, '-', linewidth=2, label=f't = {t}')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('u (偏移显示)', fontsize=11)
    ax2.set_title('抛物型: 热方程\n∂u/∂t = α ∂²u/∂x²', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.text(0.5, -0.15, '特征: 扩散,\n尖锐特征被平滑',
            fontsize=9, ha='center', transform=ax2.transAxes)

    # 椭圆型：Laplace方程
    ax3 = fig.add_subplot(1, 3, 3)
    # 边界条件决定内部解
    theta = np.linspace(0, 2*np.pi, 100)
    r_vals = [0.3, 0.5, 0.7, 0.9]
    for r in r_vals:
        # 边界u = sin(θ)的调和函数解 u(r,θ) = r*sin(θ)
        u = r * np.sin(theta)
        ax3.plot(r * np.cos(theta), r * np.sin(theta), '-', linewidth=1.5,
                label=f'r = {r}' if r in [0.3, 0.9] else None)
        # 等势线颜色
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title('椭圆型: Laplace方程\n∇²u = 0', fontsize=11)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=9)
    ax3.text(0.5, -0.15, '特征: 稳态,\n边界决定内部',
            fontsize=9, ha='center', transform=ax3.transAxes)

    plt.suptitle('三类基本PDE的特征对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figures/chap05"

    print("=" * 60)
    print("Generating continuum mechanics PDE figures...")
    print("=" * 60)

    print("\n[1] 波动方程可视化")
    plot_wave_equation(f"{output_dir}/wave_equation_visualization.pdf")

    print("\n[2] 热传导方程可视化")
    plot_heat_equation(f"{output_dir}/heat_equation_visualization.pdf")

    print("\n[3] Poisson方程可视化")
    plot_poisson_equation(f"{output_dir}/poisson_equation_visualization.pdf")

    print("\n[4] 从离散到连续")
    plot_discrete_to_continuum(f"{output_dir}/discrete_to_continuum.pdf")

    print("\n[5] 三类PDE对比")
    plot_pde_comparison(f"{output_dir}/pde_comparison.pdf")

    print("\n" + "=" * 60)
    print("All continuum mechanics figures generated!")
    print("=" * 60)
