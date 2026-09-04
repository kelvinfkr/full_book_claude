"""
第5章：经典力学 - 可视化演示
==============================

从仓库根目录运行::

    python3 code/chap05_mechanics.py          # 只重绘 figs_chap05/chap05_fig2
    python3 code/chap05_mechanics.py --all    # 同时重绘双摆混沌图 chap05_fig1

包含：
1. 双摆混沌：初值敏感性演示（chap05_fig1）
2. 单摆：小角度近似的有效范围（chap05_fig2）
   - (a)(b) θ(t)：精确方程与线性化方程的数值解
   - (c) 周期误差：周期由数值积分的过零点事件测出，并与椭圆积分公式对照
   - (d) 相图：精确能量曲线 vs 线性化的椭圆
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

setup_style()

G, L = 9.8, 1.0          # 重力加速度、摆长
T0 = 2 * np.pi * np.sqrt(L / G)   # 小角度周期 ≈ 2.007 s


# ---------------------------------------------------------------------------
# 1. 双摆混沌（保留原实现，仅修正输出路径）
# ---------------------------------------------------------------------------
def double_pendulum_deriv(t, state, L1, L2, m1, m2, g):
    """双摆运动方程（拉格朗日推导）。state = [θ1, ω1, θ2, ω2]"""
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1
    num1 = (m2 * L1 * omega1 ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            + m2 * L2 * omega2 ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1))
    num2 = (-m2 * L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * (g * np.sin(theta1) * np.cos(delta)
                           - L1 * omega1 ** 2 * np.sin(delta)
                           - g * np.sin(theta2)))
    return [omega1, num1 / den1, omega2, num2 / den2]


def create_double_pendulum_chaos():
    """双摆混沌演示：微小初值差异导致轨迹完全分化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    L1, L2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.8
    theta1_0, theta2_0 = np.pi / 2, np.pi / 2
    eps = 0.001
    state0_a = [theta1_0, 0, theta2_0, 0]
    state0_b = [theta1_0 + eps, 0, theta2_0, 0]
    t_eval = np.linspace(0, 20, 2000)
    sol_a = solve_ivp(double_pendulum_deriv, (0, 20), state0_a,
                      args=(L1, L2, m1, m2, g), t_eval=t_eval, method='RK45')
    sol_b = solve_ivp(double_pendulum_deriv, (0, 20), state0_b,
                      args=(L1, L2, m1, m2, g), t_eval=t_eval, method='RK45')

    ax1 = axes[0, 0]
    ax1.plot(sol_a.t, sol_a.y[0], 'b-', linewidth=1, label=f'θ₁(0) = {theta1_0:.3f}', alpha=0.8)
    ax1.plot(sol_b.t, sol_b.y[0], 'r-', linewidth=1, label=f'θ₁(0) = {theta1_0 + eps:.3f}', alpha=0.8)
    ax1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax1.set_ylabel('θ₁ (弧度)', fontsize=12, labelpad=10)
    ax1.set_title('双摆: θ₁随时间变化', fontsize=14)
    ax1.legend(fontsize=10)
    diff = np.abs(sol_a.y[0] - sol_b.y[0])
    diverge_idx = np.where(diff > 0.5)[0]
    if len(diverge_idx) > 0:
        t_div = sol_a.t[diverge_idx[0]]
        ax1.axvline(x=t_div, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(f'发散点\n≈{t_div:.1f}秒', xy=(t_div, 0), xytext=(t_div + 1, 2),
                     fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
        print(f'双摆：|Δθ1| 首次超过 0.5 rad 的时刻 t ≈ {t_div:.2f} s')

    ax2 = axes[0, 1]
    ax2.plot(sol_a.y[0], sol_a.y[1], 'b-', linewidth=0.5, alpha=0.7, label='轨迹 A')
    ax2.plot(sol_b.y[0], sol_b.y[1], 'r-', linewidth=0.5, alpha=0.7, label='轨迹 B')
    ax2.set_xlabel('θ₁ (弧度)', fontsize=12, labelpad=10)
    ax2.set_ylabel('ω₁ (弧度/秒)', fontsize=12, labelpad=10)
    ax2.set_title('相空间 (θ₁, ω₁)', fontsize=14)
    ax2.legend(fontsize=10)

    ax3 = axes[1, 0]
    ax3.semilogy(sol_a.t, np.abs(sol_a.y[0] - sol_b.y[0]), 'b-', linewidth=1.5, label='|Δθ₁|')
    ax3.semilogy(sol_a.t, np.abs(sol_a.y[2] - sol_b.y[2]), 'r-', linewidth=1.5, label='|Δθ₂|')
    ax3.axhline(y=eps, color='gray', linestyle='--', label=f'初始差异 = {eps}')
    ax3.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax3.set_ylabel('差异 (弧度)', fontsize=12, labelpad=10)
    ax3.set_title('指数发散 (混沌特征)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.set_ylim(1e-6, 10)

    ax4 = axes[1, 1]

    def get_positions(sol):
        th1, th2 = sol.y[0], sol.y[2]
        x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
        return x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)

    x2_a, y2_a = get_positions(sol_a)
    x2_b, y2_b = get_positions(sol_b)
    n_plot = 500
    ax4.plot(x2_a[:n_plot], y2_a[:n_plot], 'b-', linewidth=0.5, alpha=0.7, label='轨迹 A')
    ax4.plot(x2_b[:n_plot], y2_b[:n_plot], 'r-', linewidth=0.5, alpha=0.7, label='轨迹 B')
    ax4.plot(0, 0, 'ko', markersize=8)
    ax4.set_xlabel('x (米)', fontsize=12, labelpad=10)
    ax4.set_ylabel('y (米)', fontsize=12, labelpad=10)
    ax4.set_title('摆端轨迹 (前5秒)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.set_aspect('equal')

    plt.tight_layout()
    fig.savefig('figs_chap05/chap05_fig1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('已保存 figs_chap05/chap05_fig1.png')


# ---------------------------------------------------------------------------
# 2. 单摆：小角度近似的有效范围
# ---------------------------------------------------------------------------
def pendulum_exact(t, s):
    """精确方程 θ'' = -(g/L) sin θ"""
    return [s[1], -(G / L) * np.sin(s[0])]


def pendulum_linear(t, s):
    """线性化方程 θ'' = -(g/L) θ"""
    return [s[1], -(G / L) * s[0]]


def measure_period(theta0, n_periods=3):
    """用数值积分测周期：记录 θ 向下穿过 0 的时刻（事件检测），相邻事件之差即周期。

    第一次向下过零发生在 T/4，第二次在 5T/4，……因此不能用 2×(首次过零时刻)。
    """
    def downward_zero(t, s):
        return s[0]
    downward_zero.direction = -1     # 只记录 θ 由正变负的时刻
    sol = solve_ivp(pendulum_exact, (0, (n_periods + 1) * 3.0 * T0), [theta0, 0.0],
                    events=downward_zero, rtol=1e-10, atol=1e-12, dense_output=False)
    t_ev = sol.t_events[0]
    periods = np.diff(t_ev)
    return periods.mean(), t_ev


def period_formula(theta0):
    """精确周期的椭圆积分公式 T = (2/π) K(k²) T0, k = sin(θ0/2)"""
    k = np.sin(theta0 / 2)
    return T0 * 2 / np.pi * ellipk(k ** 2)


def create_simple_pendulum_comparison():
    fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2, aspect=0.66))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---------- (a)(b) θ(t)：精确 vs 线性化 ----------
    t_end = 10.0
    t_eval = np.linspace(0, t_end, 1500)
    for ax, deg, lab in ((ax_a, 30, '(a)'), (ax_b, 90, '(b)')):
        th0 = np.radians(deg)
        s_ex = solve_ivp(pendulum_exact, (0, t_end), [th0, 0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
        s_li = solve_ivp(pendulum_linear, (0, t_end), [th0, 0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
        ax.plot(s_ex.t, np.degrees(s_ex.y[0]), color=COLORS['black'], label='精确方程 $\\ddot\\theta=-(g/L)\\sin\\theta$（数值积分）')
        ax.plot(s_li.t, np.degrees(s_li.y[0]), color=COLORS['blue'], ls='--', label='小角度近似 $\\ddot\\theta=-(g/L)\\theta$')
        ax.set_xlabel('时间 $t$ (s)')
        ax.set_ylabel('摆角 $\\theta$ (°)')
        ax.set_xlim(0, t_end)
        ax.set_ylim(-deg * 1.45, deg * 1.45)
        ax.text(0.02, 0.96, f'$\\theta_0={deg}°$', transform=ax.transAxes, va='top', ha='left')
        panel_label(ax, lab)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)

    # ---------- (c) 周期误差 ----------
    theta_deg = np.arange(5, 176, 5)
    T_meas = np.array([measure_period(np.radians(d))[0] for d in theta_deg])
    err_meas = (T_meas / T0 - 1) * 100
    theta_fine = np.radians(np.linspace(1, 178, 400))
    err_formula = (period_formula(theta_fine) / T0 - 1) * 100
    err_series = theta_fine ** 2 / 16 * 100          # T ≈ T0 (1 + θ0²/16)

    ax_c.plot(np.degrees(theta_fine), err_formula, color=COLORS['black'], label='椭圆积分公式（精确周期）')
    ax_c.plot(theta_deg, err_meas, 'o', color=COLORS['blue'], ms=3.5, label='数值积分（由过零时刻测得）')
    ax_c.plot(np.degrees(theta_fine), err_series, color=COLORS['gray'], ls=':', label='二阶修正 $T_0(1+\\theta_0^2/16)$')
    ax_c.axhline(0, color=COLORS['gray'], lw=0.8)
    for d, xy_text in ((30, (8, 27)), (90, (104, 8))):
        e = (period_formula(np.radians(d)) / T0 - 1) * 100
        ax_c.annotate(f'$\\theta_0={d}°$: +{e:.1f}%', xy=(d, e), xytext=xy_text, ha='left',
                      arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8), fontsize=8.5)
    ax_c.set_xlabel('初始角 $\\theta_0$ (°)')
    ax_c.set_ylabel('周期偏差 (%)')
    ax_c.set_xlim(0, 180)
    ax_c.set_ylim(-3, 85)
    ax_c.legend(loc='upper left', fontsize=7.5)
    panel_label(ax_c, '(c)')

    # ---------- (d) 相图：精确能量曲线 vs 线性化椭圆 ----------
    th = np.linspace(-np.pi, np.pi, 800)
    first = True
    for deg in (30, 60, 90, 120):
        th0 = np.radians(deg)
        w2 = 2 * G / L * (np.cos(th) - np.cos(th0))       # 精确：ω² = 2g/L (cosθ - cosθ0)
        w_ex = np.sqrt(np.clip(w2, 0, None))
        mask = w2 >= 0
        ax_d.plot(np.degrees(th[mask]), w_ex[mask], color=COLORS['black'], lw=1.2,
                  label='精确能量曲线' if first else None)
        ax_d.plot(np.degrees(th[mask]), -w_ex[mask], color=COLORS['black'], lw=1.2)
        w_li = np.sqrt(np.clip(G / L * (th0 ** 2 - th ** 2), 0, None))   # 线性化：椭圆
        m2 = np.abs(th) <= th0
        ax_d.plot(np.degrees(th[m2]), w_li[m2], color=COLORS['blue'], ls='--', lw=1.2,
                  label='小角度近似（椭圆）' if first else None)
        ax_d.plot(np.degrees(th[m2]), -w_li[m2], color=COLORS['blue'], ls='--', lw=1.2)
        ax_d.text(deg + 3, 0.35, f'{deg}°', fontsize=6.5, ha='left', va='bottom', color=COLORS['gray'])
        first = False
    w_sep = np.sqrt(2 * G / L * (1 + np.cos(th)))          # 分界线 E = 2g/L
    ax_d.plot(np.degrees(th), w_sep, color=COLORS['red'], lw=1.0, ls='-.', label='分界线（$\\theta_0=180°$）')
    ax_d.plot(np.degrees(th), -w_sep, color=COLORS['red'], lw=1.0, ls='-.')
    ax_d.set_xlabel('摆角 $\\theta$ (°)')
    ax_d.set_ylabel('角速度 $\\omega$ (rad/s)')
    ax_d.set_xlim(-180, 180)
    ax_d.set_ylim(-7.2, 11.5)
    ax_d.set_xticks([-180, -90, 0, 90, 180])
    ax_d.legend(loc='upper left', ncol=2, fontsize=7, columnspacing=0.8, handlelength=1.6)
    panel_label(ax_d, '(d)')

    fig.tight_layout(w_pad=2.0, h_pad=1.6, rect=(0, 0, 1, 0.95))
    save_figure(fig, 'figs_chap05/chap05_fig2')

    # 控制台输出：供 caption 引用的数值
    for d in (10, 30, 60, 90):
        Tm, _ = measure_period(np.radians(d))
        print(f'θ0={d:3d}°: 测得周期 {Tm:.4f} s, 公式 {period_formula(np.radians(d)):.4f} s, '
              f'偏差 {100 * (Tm / T0 - 1):.2f}%')


if __name__ == '__main__':
    print('=== 第5章：经典力学可视化 ===')
    print('单摆小角度近似图 (chap05_fig2)...')
    create_simple_pendulum_comparison()
    if '--all' in sys.argv:
        print('双摆混沌图 (chap05_fig1)...')
        create_double_pendulum_chaos()
    print('完成。')
