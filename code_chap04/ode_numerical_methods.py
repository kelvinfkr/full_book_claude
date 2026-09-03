"""
第4章扩展：ODE数值方法与保能量积分器
比较Euler、RK4、辛积分器在简谐振子和单摆问题上的表现
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# ODE积分器实现
# =============================================================================

def forward_euler(f, y0, t_span, dt):
    """前向欧拉法（显式）"""
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = y[i] + dt * f(t[i], y[i])
    return t, y

def rk4(f, y0, t_span, dt):
    """4阶龙格-库塔法"""
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt/2 * k1)
        k3 = f(t[i] + dt/2, y[i] + dt/2 * k2)
        k4 = f(t[i] + dt, y[i] + dt * k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

def symplectic_euler(dH_dq, dH_dp, q0, p0, t_span, dt):
    """辛欧拉法（半隐式）
    适用于哈密顿系统: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    """
    t = np.arange(t_span[0], t_span[1], dt)
    q = np.zeros(len(t))
    p = np.zeros(len(t))
    q[0], p[0] = q0, p0
    for i in range(len(t)-1):
        p[i+1] = p[i] - dt * dH_dq(q[i], p[i])  # 先更新p
        q[i+1] = q[i] + dt * dH_dp(q[i], p[i+1])  # 再更新q（用新的p）
    return t, q, p

def stormer_verlet(dH_dq, dH_dp, q0, p0, t_span, dt):
    """Störmer-Verlet法（辛积分器，2阶精度）
    也称为leapfrog方法
    """
    t = np.arange(t_span[0], t_span[1], dt)
    q = np.zeros(len(t))
    p = np.zeros(len(t))
    q[0], p[0] = q0, p0
    for i in range(len(t)-1):
        p_half = p[i] - dt/2 * dH_dq(q[i], p[i])  # 半步动量
        q[i+1] = q[i] + dt * dH_dp(q[i], p_half)  # 全步位置
        p[i+1] = p_half - dt/2 * dH_dq(q[i+1], p_half)  # 完成动量
    return t, q, p

# =============================================================================
# 测试问题：简谐振子
# =============================================================================

def harmonic_oscillator_rhs(t, y, omega=1.0):
    """简谐振子: d²x/dt² = -ω²x
    写成一阶系统: y = [x, v], dy/dt = [v, -ω²x]
    """
    return np.array([y[1], -omega**2 * y[0]])

def harmonic_energy(q, p, omega=1.0):
    """简谐振子能量: H = p²/2 + ω²q²/2"""
    return 0.5 * p**2 + 0.5 * omega**2 * q**2

# =============================================================================
# 测试问题：单摆（非线性）
# =============================================================================

def pendulum_rhs(t, y, g=9.8, L=1.0):
    """单摆: d²θ/dt² = -(g/L)sin(θ)
    写成一阶系统: y = [θ, ω], dy/dt = [ω, -(g/L)sin(θ)]
    """
    return np.array([y[1], -(g/L) * np.sin(y[0])])

def pendulum_energy(theta, omega, g=9.8, L=1.0, m=1.0):
    """单摆能量: H = (1/2)mL²ω² + mgL(1-cos(θ))"""
    return 0.5 * m * L**2 * omega**2 + m * g * L * (1 - np.cos(theta))

# =============================================================================
# 生成比较图
# =============================================================================

def plot_harmonic_comparison(save_path):
    """比较不同积分器在简谐振子上的表现"""
    omega = 1.0
    y0 = np.array([1.0, 0.0])  # 初始: x=1, v=0
    t_span = (0, 50)
    dt = 0.1

    # 解析解
    t_exact = np.linspace(0, 50, 1000)
    x_exact = np.cos(omega * t_exact)

    # 数值解
    t_euler, y_euler = forward_euler(lambda t, y: harmonic_oscillator_rhs(t, y, omega), y0, t_span, dt)
    t_rk4, y_rk4 = rk4(lambda t, y: harmonic_oscillator_rhs(t, y, omega), y0, t_span, dt)

    # 辛积分器
    dH_dq = lambda q, p: omega**2 * q
    dH_dp = lambda q, p: p
    t_symp, q_symp, p_symp = symplectic_euler(dH_dq, dH_dp, 1.0, 0.0, t_span, dt)
    t_verlet, q_verlet, p_verlet = stormer_verlet(dH_dq, dH_dp, 1.0, 0.0, t_span, dt)

    # 计算能量
    E0 = harmonic_energy(1.0, 0.0, omega)
    E_euler = harmonic_energy(y_euler[:, 0], y_euler[:, 1], omega)
    E_rk4 = harmonic_energy(y_rk4[:, 0], y_rk4[:, 1], omega)
    E_symp = harmonic_energy(q_symp, p_symp, omega)
    E_verlet = harmonic_energy(q_verlet, p_verlet, omega)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 左上：轨迹对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_exact, x_exact, 'k-', linewidth=2, label='精确解', alpha=0.8)
    ax1.plot(t_euler, y_euler[:, 0], 'r--', linewidth=1.5, label='前向Euler', alpha=0.7)
    ax1.plot(t_rk4, y_rk4[:, 0], 'b-.', linewidth=1.5, label='RK4', alpha=0.7)
    ax1.plot(t_verlet, q_verlet, 'g:', linewidth=2, label='Verlet (辛积分器)', alpha=0.7)
    ax1.set_xlabel('时间 t', fontsize=11)
    ax1.set_ylabel('位置 x', fontsize=11)
    ax1.set_title('简谐振子：轨迹对比', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    # 右上：能量对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhline(y=E0, color='k', linestyle='-', linewidth=2, label='精确值 (常数)', alpha=0.8)
    ax2.plot(t_euler, E_euler, 'r-', linewidth=1.5, label='前向Euler', alpha=0.7)
    ax2.plot(t_rk4, E_rk4, 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    ax2.plot(t_verlet, E_verlet, 'g-', linewidth=1.5, label='Verlet', alpha=0.7)
    ax2.set_xlabel('时间 t', fontsize=11)
    ax2.set_ylabel('能量 H', fontsize=11)
    ax2.set_title('能量守恒对比', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)

    # 左下：相空间轨迹
    ax3 = fig.add_subplot(gs[1, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='精确 (圆)', alpha=0.8)
    ax3.plot(y_euler[:, 0], y_euler[:, 1], 'r-', linewidth=1, label='Euler (发散)', alpha=0.6)
    ax3.plot(y_rk4[:, 0], y_rk4[:, 1], 'b-', linewidth=1, label='RK4 (漂移)', alpha=0.6)
    ax3.plot(q_verlet, p_verlet, 'g-', linewidth=1, label='Verlet (保持)', alpha=0.6)
    ax3.set_xlabel('位置 x', fontsize=11)
    ax3.set_ylabel('速度 v', fontsize=11)
    ax3.set_title('相空间轨迹', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)

    # 右下：相对能量误差
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t_euler, np.abs(E_euler - E0)/E0 + 1e-16, 'r-', linewidth=1.5, label='前向Euler', alpha=0.7)
    ax4.semilogy(t_rk4, np.abs(E_rk4 - E0)/E0 + 1e-16, 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    ax4.semilogy(t_verlet, np.abs(E_verlet - E0)/E0 + 1e-16, 'g-', linewidth=1.5, label='Verlet', alpha=0.7)
    ax4.set_xlabel('时间 t', fontsize=11)
    ax4.set_ylabel('相对能量误差 |E-E0|/E0', fontsize=11)
    ax4.set_title('能量误差增长', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pendulum_comparison(save_path):
    """比较不同积分器在单摆问题上的表现"""
    g, L = 9.8, 1.0
    theta0, omega0 = 0.5, 0.0  # 初始角度0.5rad，角速度0
    t_span = (0, 20)
    dt = 0.05

    # 数值解
    y0 = np.array([theta0, omega0])
    t_euler, y_euler = forward_euler(lambda t, y: pendulum_rhs(t, y, g, L), y0, t_span, dt)
    t_rk4, y_rk4 = rk4(lambda t, y: pendulum_rhs(t, y, g, L), y0, t_span, dt)

    # 辛积分器
    dH_dq = lambda q, p: g/L * np.sin(q)  # ∂H/∂θ = (g/L)sin(θ)
    dH_dp = lambda q, p: p  # ∂H/∂ω = ω
    t_verlet, q_verlet, p_verlet = stormer_verlet(dH_dq, dH_dp, theta0, omega0, t_span, dt)

    # 计算能量
    E0 = pendulum_energy(theta0, omega0, g, L)
    E_euler = pendulum_energy(y_euler[:, 0], y_euler[:, 1], g, L)
    E_rk4 = pendulum_energy(y_rk4[:, 0], y_rk4[:, 1], g, L)
    E_verlet = pendulum_energy(q_verlet, p_verlet, g, L)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 左：角度随时间变化
    axes[0].plot(t_euler, y_euler[:, 0], 'r-', linewidth=1.5, label='Euler', alpha=0.7)
    axes[0].plot(t_rk4, y_rk4[:, 0], 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    axes[0].plot(t_verlet, q_verlet, 'g-', linewidth=1.5, label='Verlet', alpha=0.7)
    axes[0].set_xlabel('时间 t (s)', fontsize=11)
    axes[0].set_ylabel('角度 θ (rad)', fontsize=11)
    axes[0].set_title('单摆：角度随时间变化', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 中：能量对比
    axes[1].axhline(y=E0, color='k', linestyle='--', linewidth=2, label='初始能量 E0', alpha=0.8)
    axes[1].plot(t_euler, E_euler, 'r-', linewidth=1.5, label='Euler', alpha=0.7)
    axes[1].plot(t_rk4, E_rk4, 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    axes[1].plot(t_verlet, E_verlet, 'g-', linewidth=1.5, label='Verlet', alpha=0.7)
    axes[1].set_xlabel('时间 t (s)', fontsize=11)
    axes[1].set_ylabel('能量 (J)', fontsize=11)
    axes[1].set_title('能量守恒对比', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 右：相空间
    axes[2].plot(y_euler[:, 0], y_euler[:, 1], 'r-', linewidth=1, label='Euler', alpha=0.6)
    axes[2].plot(y_rk4[:, 0], y_rk4[:, 1], 'b-', linewidth=1, label='RK4', alpha=0.6)
    axes[2].plot(q_verlet, p_verlet, 'g-', linewidth=1, label='Verlet', alpha=0.6)
    axes[2].set_xlabel('角度 θ (rad)', fontsize=11)
    axes[2].set_ylabel('角速度 ω (rad/s)', fontsize=11)
    axes[2].set_title('相空间轨迹', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_long_time_comparison(save_path):
    """长时间积分对比：辛积分器的优势"""
    omega = 1.0
    y0 = np.array([1.0, 0.0])
    t_span = (0, 500)  # 长时间
    dt = 0.1

    # RK4
    t_rk4, y_rk4 = rk4(lambda t, y: harmonic_oscillator_rhs(t, y, omega), y0, t_span, dt)
    E_rk4 = harmonic_energy(y_rk4[:, 0], y_rk4[:, 1], omega)

    # Verlet
    dH_dq = lambda q, p: omega**2 * q
    dH_dp = lambda q, p: p
    t_verlet, q_verlet, p_verlet = stormer_verlet(dH_dq, dH_dp, 1.0, 0.0, t_span, dt)
    E_verlet = harmonic_energy(q_verlet, p_verlet, omega)

    E0 = harmonic_energy(1.0, 0.0, omega)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 能量漂移
    ax1.plot(t_rk4, (E_rk4 - E0), 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    ax1.plot(t_verlet, (E_verlet - E0), 'g-', linewidth=1.5, label='Verlet (辛积分器)', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('时间 t', fontsize=11)
    ax1.set_ylabel('能量漂移 (E - E0)', fontsize=11)
    ax1.set_title('长时间能量漂移 (t=0 到 500)', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 添加RK4注释（位置调整）
    rk4_drift_at_400 = E_rk4[4000] - E0
    ax1.annotate('RK4: 系统性漂移\n(随时间累积)',
                xy=(400, rk4_drift_at_400), fontsize=9,
                xytext=(420, rk4_drift_at_400 * 0.6),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.8))

    # 放大看Verlet的振荡
    ax2.plot(t_verlet, (E_verlet - E0), 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('时间 t', fontsize=11)
    ax2.set_ylabel('能量漂移 (E - E0)', fontsize=11)
    ax2.set_title('Verlet: 有界能量振荡', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加Verlet注释（位置调整）
    verlet_max = np.max(np.abs(E_verlet - E0))
    ax2.annotate('辛积分器: 有界振荡\n(无系统性漂移)',
                xy=(250, verlet_max * 0.5), fontsize=9,
                xytext=(280, verlet_max * 0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_convergence_order(save_path):
    """验证各方法的收敛阶"""
    omega = 1.0
    y0 = np.array([1.0, 0.0])
    t_final = 1.0

    dts = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    errors_euler = []
    errors_rk4 = []
    errors_verlet = []

    # 解析解
    x_exact = np.cos(omega * t_final)

    for dt in dts:
        t_span = (0, t_final + dt)

        # Euler
        _, y = forward_euler(lambda t, y: harmonic_oscillator_rhs(t, y, omega), y0, t_span, dt)
        errors_euler.append(np.abs(y[-1, 0] - x_exact))

        # RK4
        _, y = rk4(lambda t, y: harmonic_oscillator_rhs(t, y, omega), y0, t_span, dt)
        errors_rk4.append(np.abs(y[-1, 0] - x_exact))

        # Verlet
        dH_dq = lambda q, p: omega**2 * q
        dH_dp = lambda q, p: p
        _, q, _ = stormer_verlet(dH_dq, dH_dp, 1.0, 0.0, t_span, dt)
        errors_verlet.append(np.abs(q[-1] - x_exact))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(dts, errors_euler, 'ro-', markersize=8, linewidth=2, label='Euler (1阶)')
    ax.loglog(dts, errors_rk4, 'bs-', markersize=8, linewidth=2, label='RK4 (4阶)')
    ax.loglog(dts, errors_verlet, 'g^-', markersize=8, linewidth=2, label='Verlet (2阶)')

    # 参考线
    ax.loglog(dts, 0.5*dts, 'r--', alpha=0.5, label='O(Δt)')
    ax.loglog(dts, 0.1*dts**2, 'g--', alpha=0.5, label='O(Δt²)')
    ax.loglog(dts, 0.01*dts**4, 'b--', alpha=0.5, label='O(Δt⁴)')

    ax.set_xlabel('步长 Δt', fontsize=12)
    ax.set_ylabel('误差 |x(1) - x精确|', fontsize=12)
    ax.set_title('收敛阶验证', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap04"

    print("=" * 60)
    print("Generating ODE numerical methods figures...")
    print("=" * 60)

    print("\n[1] Harmonic oscillator comparison")
    plot_harmonic_comparison(f"{output_dir}/harmonic_comparison.pdf")

    print("\n[2] Pendulum comparison")
    plot_pendulum_comparison(f"{output_dir}/pendulum_comparison.pdf")

    print("\n[3] Long-time integration")
    plot_long_time_comparison(f"{output_dir}/long_time_comparison.pdf")

    print("\n[4] Convergence order verification")
    plot_convergence_order(f"{output_dir}/convergence_order.pdf")

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)
