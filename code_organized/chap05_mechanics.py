"""
第5章：经典力学 - 可视化演示

包含：
1. 双摆混沌：初值敏感性演示
2. 单摆：小角度近似 vs 精确解
3. 拉格朗日量与广义坐标
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()


def double_pendulum_deriv(t, state, L1, L2, m1, m2, g):
    """
    双摆运动方程（拉格朗日推导）

    state = [θ1, ω1, θ2, ω2]
    """
    theta1, omega1, theta2, omega2 = state

    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    # θ1的加速度
    num1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
            m2 * g * np.sin(theta2) * np.cos(delta) +
            m2 * L2 * omega2**2 * np.sin(delta) -
            (m1 + m2) * g * np.sin(theta1))
    alpha1 = num1 / den1

    # θ2的加速度
    num2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
            (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) -
                         L1 * omega1**2 * np.sin(delta) -
                         g * np.sin(theta2)))
    alpha2 = num2 / den2

    return [omega1, alpha1, omega2, alpha2]


def create_double_pendulum_chaos():
    """
    双摆混沌演示：微小初值差异导致轨迹完全分化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 参数
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.8

    # 初始条件
    theta1_0 = np.pi / 2  # 90度
    theta2_0 = np.pi / 2
    omega1_0 = 0
    omega2_0 = 0

    # 两条轨迹，初值仅差0.001弧度（约0.057度）
    eps = 0.001
    state0_a = [theta1_0, omega1_0, theta2_0, omega2_0]
    state0_b = [theta1_0 + eps, omega1_0, theta2_0, omega2_0]

    # 积分时间
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)

    sol_a = solve_ivp(double_pendulum_deriv, t_span, state0_a,
                      args=(L1, L2, m1, m2, g), t_eval=t_eval, method='RK45')
    sol_b = solve_ivp(double_pendulum_deriv, t_span, state0_b,
                      args=(L1, L2, m1, m2, g), t_eval=t_eval, method='RK45')

    # === 子图1: θ1随时间变化 ===
    ax1 = axes[0, 0]
    ax1.plot(sol_a.t, sol_a.y[0], 'b-', linewidth=1, label=f'θ₁(0) = {theta1_0:.3f}', alpha=0.8)
    ax1.plot(sol_b.t, sol_b.y[0], 'r-', linewidth=1, label=f'θ₁(0) = {theta1_0 + eps:.3f}', alpha=0.8)
    ax1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax1.set_ylabel('θ₁ (弧度)', fontsize=12, labelpad=10)
    ax1.set_title('双摆: θ₁随时间变化', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 标记分岔点
    diff = np.abs(sol_a.y[0] - sol_b.y[0])
    diverge_idx = np.where(diff > 0.5)[0]
    if len(diverge_idx) > 0:
        t_div = sol_a.t[diverge_idx[0]]
        ax1.axvline(x=t_div, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(f'发散点\n≈{t_div:.1f}秒', xy=(t_div, 0),
                     xytext=(t_div + 1, 2), fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='gray'))

    # === 子图2: 相空间轨迹 ===
    ax2 = axes[0, 1]
    ax2.plot(sol_a.y[0], sol_a.y[1], 'b-', linewidth=0.5, alpha=0.7, label='轨迹 A')
    ax2.plot(sol_b.y[0], sol_b.y[1], 'r-', linewidth=0.5, alpha=0.7, label='轨迹 B')
    ax2.set_xlabel('θ₁ (弧度)', fontsize=12, labelpad=10)
    ax2.set_ylabel('ω₁ (弧度/秒)', fontsize=12, labelpad=10)
    ax2.set_title('相空间 (θ₁, ω₁)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === 子图3: 轨迹差异随时间增长 ===
    ax3 = axes[1, 0]
    diff_theta1 = np.abs(sol_a.y[0] - sol_b.y[0])
    diff_theta2 = np.abs(sol_a.y[2] - sol_b.y[2])

    ax3.semilogy(sol_a.t, diff_theta1, 'b-', linewidth=1.5, label='|Δθ₁|')
    ax3.semilogy(sol_a.t, diff_theta2, 'r-', linewidth=1.5, label='|Δθ₂|')
    ax3.axhline(y=eps, color='gray', linestyle='--', label=f'初始差异 = {eps}')

    ax3.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax3.set_ylabel('差异 (弧度)', fontsize=12, labelpad=10)
    ax3.set_title('指数发散 (混沌特征)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-6, 10)

    # === 子图4: 双摆端点轨迹 ===
    ax4 = axes[1, 1]

    def get_positions(sol, L1, L2):
        theta1, theta2 = sol.y[0], sol.y[2]
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        return x1, y1, x2, y2

    x1_a, y1_a, x2_a, y2_a = get_positions(sol_a, L1, L2)
    x1_b, y1_b, x2_b, y2_b = get_positions(sol_b, L1, L2)

    # 只画前500个点，避免太乱
    n_plot = 500
    ax4.plot(x2_a[:n_plot], y2_a[:n_plot], 'b-', linewidth=0.5, alpha=0.7, label='轨迹 A')
    ax4.plot(x2_b[:n_plot], y2_b[:n_plot], 'r-', linewidth=0.5, alpha=0.7, label='轨迹 B')

    ax4.plot(0, 0, 'ko', markersize=8)  # 悬挂点
    ax4.set_xlabel('x (米)', fontsize=12, labelpad=10)
    ax4.set_ylabel('y (米)', fontsize=12, labelpad=10)
    ax4.set_title('摆端轨迹 (前5秒)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap05_fig1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap05_fig1.png")


def create_simple_pendulum_comparison():
    """
    单摆：小角度近似 vs 精确解
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    g = 9.8
    L = 1.0

    def pendulum_exact(t, state):
        """精确方程：θ'' = -(g/L)sin(θ)"""
        theta, omega = state
        return [omega, -(g/L) * np.sin(theta)]

    def pendulum_approx(t, state):
        """近似方程：θ'' = -(g/L)θ"""
        theta, omega = state
        return [omega, -(g/L) * theta]

    # 不同初始角度
    theta0_list = [0.1, 0.3, 0.5, 1.0]  # 弧度
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)

    # === 子图1: 不同初始角度的θ(t) ===
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(theta0_list)))

    for theta0, color in zip(theta0_list, colors):
        state0 = [theta0, 0]

        sol_exact = solve_ivp(pendulum_exact, t_span, state0, t_eval=t_eval)
        sol_approx = solve_ivp(pendulum_approx, t_span, state0, t_eval=t_eval)

        ax1.plot(sol_exact.t, sol_exact.y[0], '-', color=color, linewidth=1.5,
                 label=f'θ₀={np.degrees(theta0):.0f}° 精确')
        ax1.plot(sol_approx.t, sol_approx.y[0], '--', color=color, linewidth=1.5,
                 alpha=0.7)

    ax1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax1.set_ylabel('θ (弧度)', fontsize=12, labelpad=10)
    ax1.set_title('单摆: 精确解(实线) vs 近似解(虚线)', fontsize=14)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # === 子图2: 周期误差 ===
    ax2 = axes[1]

    theta0_range = np.linspace(0.01, np.pi/2, 50)
    period_exact = []
    period_approx = 2 * np.pi * np.sqrt(L / g)  # 小角度近似周期

    for theta0 in theta0_range:
        # 数值计算精确周期（找第一个过零点的时间的两倍）
        sol = solve_ivp(pendulum_exact, (0, 20), [theta0, 0],
                        t_eval=np.linspace(0, 20, 2000))
        # 找角度过零且速度为负的点
        for i in range(1, len(sol.t)):
            if sol.y[0][i-1] > 0 and sol.y[0][i] <= 0:
                # 线性插值找精确过零时间
                t_zero = sol.t[i-1] + (0 - sol.y[0][i-1]) * (sol.t[i] - sol.t[i-1]) / (sol.y[0][i] - sol.y[0][i-1])
                period_exact.append(2 * t_zero)
                break
        else:
            period_exact.append(np.nan)

    period_exact = np.array(period_exact)
    relative_error = (period_exact - period_approx) / period_approx * 100

    ax2.plot(np.degrees(theta0_range), relative_error, 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel('初始角度 θ₀ (度)', fontsize=12, labelpad=10)
    ax2.set_ylabel('周期误差 (%)', fontsize=12, labelpad=10)
    ax2.set_title('周期：小角度近似误差', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标注几个关键点
    for theta_deg in [10, 30, 60, 90]:
        theta_rad = np.radians(theta_deg)
        idx = np.argmin(np.abs(theta0_range - theta_rad))
        err = relative_error[idx]
        ax2.annotate(f'{err:.1f}%', (theta_deg, err),
                     textcoords="offset points", xytext=(5, 5), fontsize=10)

    # === 子图3: 相图对比 ===
    ax3 = axes[2]

    # 画精确解的能量等高线
    theta_grid = np.linspace(-np.pi, np.pi, 200)
    omega_grid = np.linspace(-5, 5, 200)
    Theta, Omega = np.meshgrid(theta_grid, omega_grid)

    # 能量 E = (1/2)L²ω² + gL(1-cosθ)
    E = 0.5 * L**2 * Omega**2 + g * L * (1 - np.cos(Theta))

    contours = ax3.contour(Theta, Omega, E, levels=np.linspace(0, 30, 15),
                           colors='blue', alpha=0.5)

    # 画近似解的能量等高线（椭圆）
    E_approx = 0.5 * L**2 * Omega**2 + 0.5 * g * L * Theta**2
    ax3.contour(Theta, Omega, E_approx, levels=np.linspace(0, 30, 15),
                colors='red', linestyles='--', alpha=0.5)

    ax3.set_xlabel('θ (弧度)', fontsize=12, labelpad=10)
    ax3.set_ylabel('ω (弧度/秒)', fontsize=12, labelpad=10)
    ax3.set_title('相图: 精确(蓝) vs 近似(红)', fontsize=14)
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap05_fig2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap05_fig2.png")


def create_lagrangian_demo():
    """
    拉格朗日量演示：动能-势能在运动过程中的变化
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    g = 9.8
    L = 1.0
    m = 1.0

    # 单摆运动
    theta0 = np.pi / 4
    sol = solve_ivp(lambda t, s: [s[1], -(g/L)*np.sin(s[0])],
                    (0, 10), [theta0, 0], t_eval=np.linspace(0, 10, 500))

    theta = sol.y[0]
    omega = sol.y[1]
    t = sol.t

    # 计算能量
    T = 0.5 * m * L**2 * omega**2  # 动能
    V = m * g * L * (1 - np.cos(theta))  # 势能
    E = T + V  # 总能量
    L_lag = T - V  # 拉格朗日量

    # === 子图1: 能量随时间变化 ===
    ax1 = axes[0]
    ax1.plot(t, T, 'r-', linewidth=2, label='动能 T')
    ax1.plot(t, V, 'b-', linewidth=2, label='势能 V')
    ax1.plot(t, E, 'k--', linewidth=2, label='总能量 E=T+V')
    ax1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax1.set_ylabel('能量 (焦耳)', fontsize=12, labelpad=10)
    ax1.set_title('单摆中的能量守恒', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 子图2: 拉格朗日量 ===
    ax2 = axes[1]
    ax2.plot(t, L_lag, 'g-', linewidth=2, label='L = T - V')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax2.set_ylabel('拉格朗日量 L (焦耳)', fontsize=12, labelpad=10)
    ax2.set_title('拉格朗日量 L = T - V', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 添加注释
    ax2.annotate('L > 0: T > V\n(速度快，位置低)',
                 xy=(t[np.argmax(L_lag)], np.max(L_lag)),
                 xytext=(t[np.argmax(L_lag)] + 0.5, np.max(L_lag) + 0.1),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

    ax2.annotate('L < 0: T < V\n(速度慢，位置高)',
                 xy=(t[np.argmin(L_lag)], np.min(L_lag)),
                 xytext=(t[np.argmin(L_lag)] + 0.5, np.min(L_lag) - 0.15),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap05_fig3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap05_fig3.png")


if __name__ == "__main__":
    print("=== 第5章：经典力学可视化 ===\n")

    print("1. 生成双摆混沌图...")
    create_double_pendulum_chaos()

    print("\n2. 生成单摆近似对比图...")
    create_simple_pendulum_comparison()

    print("\n3. 生成拉格朗日量演示图...")
    create_lagrangian_demo()

    print("\n所有图像生成完成!")
