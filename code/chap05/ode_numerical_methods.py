"""
第5章扩展：ODE数值方法与保能量积分器
=====================================

比较 Euler、RK4、辛积分器（辛 Euler、Störmer-Verlet）在简谐振子上的表现。

从仓库根目录运行::

    python3 code/chap05/ode_numerical_methods.py

输出（同时给出 pdf 与 png）：
- figures/chap05/harmonic_comparison   简谐振子：轨迹 / 能量 / 相空间 / 相对能量误差
- figures/chap05/long_time_comparison  长时间积分：RK4 能量线性漂移 vs Verlet 有界振荡
- figures/chap05/convergence_order     收敛阶验证（误差 vs 步长，双对数坐标 + 拟合斜率）
- figures/chap05/pendulum_comparison   单摆（非线性）三种积分器对比（正文未引用，保留）
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

setup_style()

# 三种方法在全书中的固定配色：Euler 红（发散/警示）、RK4 蓝（数值解）、Verlet 绿（保结构）
C_EULER, C_RK4, C_VERLET, C_SYMP = COLORS['red'], COLORS['blue'], COLORS['green'], COLORS['orange']


# =============================================================================
# ODE积分器实现
# =============================================================================
def time_grid(t_span, dt):
    """等距时间网格：用步数取整，避免 np.arange 的浮点尾巴问题。"""
    n = int(round((t_span[1] - t_span[0]) / dt))
    return t_span[0] + dt * np.arange(n + 1)


def forward_euler(f, y0, t_span, dt):
    """前向欧拉法（显式，1阶）"""
    t = time_grid(t_span, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + dt * f(t[i], y[i])
    return t, y


def rk4(f, y0, t_span, dt):
    """4阶龙格-库塔法"""
    t = time_grid(t_span, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt / 2, y[i] + dt / 2 * k1)
        k3 = f(t[i] + dt / 2, y[i] + dt / 2 * k2)
        k4 = f(t[i] + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def symplectic_euler(dH_dq, dH_dp, q0, p0, t_span, dt):
    """辛欧拉法（半隐式，1阶）：dq/dt = ∂H/∂p, dp/dt = -∂H/∂q"""
    t = time_grid(t_span, dt)
    q, p = np.zeros(len(t)), np.zeros(len(t))
    q[0], p[0] = q0, p0
    for i in range(len(t) - 1):
        p[i + 1] = p[i] - dt * dH_dq(q[i], p[i])
        q[i + 1] = q[i] + dt * dH_dp(q[i], p[i + 1])
    return t, q, p


def stormer_verlet(dH_dq, dH_dp, q0, p0, t_span, dt):
    """Störmer-Verlet 法（辛积分器，2阶，也称 leapfrog）"""
    t = time_grid(t_span, dt)
    q, p = np.zeros(len(t)), np.zeros(len(t))
    q[0], p[0] = q0, p0
    for i in range(len(t) - 1):
        p_half = p[i] - dt / 2 * dH_dq(q[i], p[i])
        q[i + 1] = q[i] + dt * dH_dp(q[i], p_half)
        p[i + 1] = p_half - dt / 2 * dH_dq(q[i + 1], p_half)
    return t, q, p


# =============================================================================
# 测试问题
# =============================================================================
def harmonic_oscillator_rhs(t, y, omega=1.0):
    """简谐振子 y = [x, v], dy/dt = [v, -ω²x]"""
    return np.array([y[1], -omega ** 2 * y[0]])


def harmonic_energy(q, p, omega=1.0):
    return 0.5 * p ** 2 + 0.5 * omega ** 2 * q ** 2


def pendulum_rhs(t, y, g=9.8, L=1.0):
    return np.array([y[1], -(g / L) * np.sin(y[0])])


def pendulum_energy(theta, omega, g=9.8, L=1.0, m=1.0):
    return 0.5 * m * L ** 2 * omega ** 2 + m * g * L * (1 - np.cos(theta))


def run_harmonic(omega, y0, t_span, dt):
    """在同一设置下跑四种积分器，返回 dict。"""
    f = lambda t, y: harmonic_oscillator_rhs(t, y, omega)
    dH_dq = lambda q, p: omega ** 2 * q
    dH_dp = lambda q, p: p
    out = {}
    t, y = forward_euler(f, y0, t_span, dt)
    out['Euler'] = (t, y[:, 0], y[:, 1])
    t, y = rk4(f, y0, t_span, dt)
    out['RK4'] = (t, y[:, 0], y[:, 1])
    t, q, p = symplectic_euler(dH_dq, dH_dp, y0[0], y0[1], t_span, dt)
    out['SympEuler'] = (t, q, p)
    t, q, p = stormer_verlet(dH_dq, dH_dp, y0[0], y0[1], t_span, dt)
    out['Verlet'] = (t, q, p)
    return out


# =============================================================================
# 图 1：简谐振子对比
# =============================================================================
def plot_harmonic_comparison(save_path):
    omega, y0, t_span, dt = 1.0, np.array([1.0, 0.0]), (0.0, 50.0), 0.1
    res = run_harmonic(omega, y0, t_span, dt)
    E0 = harmonic_energy(y0[0], y0[1], omega)
    t_exact = np.linspace(*t_span, 2000)

    fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2, aspect=0.68))
    ax1, ax2, ax3, ax4 = axes.ravel()
    styles = {'Euler': dict(color=C_EULER, ls='--', label='前向 Euler'),
              'RK4': dict(color=C_RK4, ls='-', label='RK4'),
              'Verlet': dict(color=C_VERLET, ls='-', label='Verlet（辛）')}

    # (a) 轨迹
    ax1.plot(t_exact, np.cos(omega * t_exact), color=COLORS['black'], lw=2.2, alpha=0.35, label='精确解')
    for k, st in styles.items():
        t, q, _ = res[k]
        ax1.plot(t, q, lw=1.2, **st)
    ax1.set_xlabel('时间 $t$')
    ax1.set_ylabel('位置 $x$')
    ax1.set_xlim(*t_span)
    ax1.set_ylim(-3, 3)
    panel_label(ax1, '(a)')

    # (b) 能量
    ax2.axhline(E0, color=COLORS['black'], lw=1.2)
    for k, st in styles.items():
        t, q, p = res[k]
        ax2.plot(t, harmonic_energy(q, p, omega), lw=1.2, **st)
    ax2.set_xlabel('时间 $t$')
    ax2.set_ylabel('能量 $H$')
    ax2.set_xlim(*t_span)
    ax2.set_ylim(0, 4)
    panel_label(ax2, '(b)')

    # (c) 相空间
    th = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(th), np.sin(th), color=COLORS['black'], lw=2.2, alpha=0.35)
    for k, st in styles.items():
        t, q, p = res[k]
        ax3.plot(q, p, lw=0.9, **st)
    ax3.set_xlabel('位置 $x$')
    ax3.set_ylabel('速度 $v$')
    ax3.set_aspect('equal')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    panel_label(ax3, '(c)')

    # (d) 相对能量误差（对数）
    for k, st in styles.items():
        t, q, p = res[k]
        err = np.abs(harmonic_energy(q, p, omega) - E0) / E0
        ax4.semilogy(t[1:], err[1:], lw=1.2, **st)
    ax4.set_xlabel('时间 $t$')
    ax4.set_ylabel('相对能量误差 $|E-E_0|/E_0$')
    ax4.set_xlim(*t_span)
    ax4.set_ylim(1e-10, 1e2)
    panel_label(ax4, '(d)')

    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(w_pad=2.0, h_pad=1.5, rect=(0, 0, 1, 0.955))
    save_figure(fig, save_path)


# =============================================================================
# 图 2：单摆（非线性）对比——正文未引用，保留旧内容
# =============================================================================
def plot_pendulum_comparison(save_path):
    g, L, theta0, omega0, t_span, dt = 9.8, 1.0, 0.5, 0.0, (0.0, 20.0), 0.05
    y0 = np.array([theta0, omega0])
    f = lambda t, y: pendulum_rhs(t, y, g, L)
    t_e, y_e = forward_euler(f, y0, t_span, dt)
    t_r, y_r = rk4(f, y0, t_span, dt)
    t_v, q_v, p_v = stormer_verlet(lambda q, p: g / L * np.sin(q), lambda q, p: p, theta0, omega0, t_span, dt)
    E0 = pendulum_energy(theta0, omega0, g, L)

    fig, axes = plt.subplots(1, 3, figsize=fig_size(3, 1, aspect=0.8))
    axes[0].plot(t_e, y_e[:, 0], color=C_EULER, ls='--', label='Euler')
    axes[0].plot(t_r, y_r[:, 0], color=C_RK4, label='RK4')
    axes[0].plot(t_v, q_v, color=C_VERLET, label='Verlet')
    axes[0].set_xlabel('时间 $t$ (s)')
    axes[0].set_ylabel('角度 $\\theta$ (rad)')
    axes[0].legend(fontsize=8)
    axes[1].axhline(E0, color=COLORS['black'], label='$E_0$')
    axes[1].plot(t_e, pendulum_energy(y_e[:, 0], y_e[:, 1], g, L), color=C_EULER, ls='--', label='Euler')
    axes[1].plot(t_r, pendulum_energy(y_r[:, 0], y_r[:, 1], g, L), color=C_RK4, label='RK4')
    axes[1].plot(t_v, pendulum_energy(q_v, p_v, g, L), color=C_VERLET, label='Verlet')
    axes[1].set_xlabel('时间 $t$ (s)')
    axes[1].set_ylabel('能量 (J)')
    axes[1].legend(fontsize=8)
    axes[2].plot(y_e[:, 0], y_e[:, 1], color=C_EULER, lw=0.8, label='Euler')
    axes[2].plot(y_r[:, 0], y_r[:, 1], color=C_RK4, lw=0.8, label='RK4')
    axes[2].plot(q_v, p_v, color=C_VERLET, lw=0.8, label='Verlet')
    axes[2].set_xlabel('角度 $\\theta$ (rad)')
    axes[2].set_ylabel('角速度 $\\omega$ (rad/s)')
    axes[2].legend(fontsize=8)
    for ax, lab in zip(axes, '(a) (b) (c)'.split()):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, save_path)


# =============================================================================
# 图 3：长时间积分——RK4 线性漂移 vs Verlet 有界振荡
# =============================================================================
def plot_long_time_comparison(save_path):
    omega, y0, t_span, dt = 1.0, np.array([1.0, 0.0]), (0.0, 500.0), 0.1
    res = run_harmonic(omega, y0, t_span, dt)
    E0 = harmonic_energy(y0[0], y0[1], omega)
    t, q_r, p_r = res['RK4']
    _, q_v, p_v = res['Verlet']
    dE_rk4 = harmonic_energy(q_r, p_r, omega) - E0
    dE_ver = harmonic_energy(q_v, p_v, omega) - E0

    # Verlet：按周期分块取包络（每周期 2π/ω 含 63 步左右）
    period_steps = int(round(2 * np.pi / omega / dt))
    n_blocks = len(t) // period_steps
    tb = t[:n_blocks * period_steps].reshape(n_blocks, period_steps).mean(axis=1)
    blk = dE_ver[:n_blocks * period_steps].reshape(n_blocks, period_steps)
    ver_max, ver_min, ver_mean = blk.max(axis=1), blk.min(axis=1), blk.mean(axis=1)
    blk_r = dE_rk4[:n_blocks * period_steps].reshape(n_blocks, period_steps)
    rk4_absmax = np.abs(blk_r).max(axis=1)

    # RK4 漂移的线性拟合（相对能量误差 vs t，双对数下拟合斜率）
    mask = tb > 5
    slope, intercept = np.polyfit(np.log(tb[mask]), np.log(rk4_absmax[mask] / E0), 1)
    ver_level = np.abs(blk).max(axis=1) / E0
    t_cross = np.exp((np.log(ver_level.mean()) - intercept) / slope)   # 外推：RK4 何时超过 Verlet 的水平

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(2, 1, aspect=0.78))

    # (a) 双对数：|E-E0|/E0 vs t
    ax1.loglog(tb, rk4_absmax / E0, color=C_RK4, label='RK4（每周期最大值）')
    ax1.loglog(tb, ver_level, color=C_VERLET, label='Verlet（每周期最大值）')
    t_ext = np.array([tb[mask][0], t_cross * 1.5])
    ax1.loglog(t_ext, np.exp(intercept) * t_ext ** slope, color=C_RK4, ls=':', lw=1.0,
               label=f'RK4 拟合与外推：$\\propto t^{{{slope:.2f}}}$')
    ax1.axvline(t_span[1], color=COLORS['gray'], lw=0.8, ls='--')
    ax1.text(t_span[1] * 1.15, 2.5e-6, '积分终点\n$t=500$', va='bottom', ha='left', fontsize=7.5, color=COLORS['gray'])
    mant, expo = f'{t_cross:.1e}'.split('e')
    ax1.annotate(f'外推交点 $t\\approx{mant}\\times10^{{{int(expo)}}}$', xy=(t_cross, ver_level.mean()),
                 xytext=(t_cross * 0.35, 3e-5), ha='center', fontsize=8,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8))
    ax1.set_xlabel('时间 $t$（对数）')
    ax1.set_ylabel('相对能量误差 $|E-E_0|/E_0$')
    ax1.set_xlim(5, t_cross * 2)
    ax1.set_ylim(1e-9, 1e-2)
    ax1.legend(loc='lower left', fontsize=7.5)
    panel_label(ax1, '(a)')

    # (b) 线性坐标放大：t∈[0,30] 的原始 E-E0
    t_zoom = 30.0
    m = t <= t_zoom
    ax2.plot(t[m], dE_ver[m] / E0, color=C_VERLET, lw=1.2, label='Verlet：$E-E_0$（逐步）')
    ax2.plot(t[m], dE_rk4[m] / E0, color=C_RK4, lw=1.4, label='RK4：$E-E_0$（逐步；此尺度下$\\approx0$）')
    ax2.axhline(ver_max.max() / E0, color=C_VERLET, ls='--', lw=0.8)
    ax2.axhline(ver_min.min() / E0, color=C_VERLET, ls='--', lw=0.8)
    ax2.text(0.5, 0.015, f'Verlet 全程（$t\\leq 500$）的上下界（虚线）：\n[{ver_min.min() / E0 * 1e3:.2f}, {ver_max.max() / E0 * 1e3:.1f}]$\\times10^{{-3}}$，有界、无漂移',
             transform=ax2.transAxes, ha='center', va='bottom', fontsize=7, color=C_VERLET)
    ax2.axhline(0, color=COLORS['black'], lw=0.8)
    ax2.set_xlabel('时间 $t$（线性，前 30 个时间单位）')
    ax2.set_ylabel('相对能量偏差 $(E-E_0)/E_0$')
    ax2.set_xlim(0, t_zoom)
    ax2.set_ylim(-3.7e-3, 1.7e-3)
    ax2.legend(loc='upper right', fontsize=7.5)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    save_figure(fig, save_path)
    print(f'  RK4: t=500 时 (E-E0)/E0 = {dE_rk4[-1] / E0:.3e}，双对数拟合斜率 {slope:.3f}，外推交点 t≈{t_cross:.0f}')
    print(f'  Verlet: 全程 (E-E0)/E0 ∈ [{ver_min.min() / E0:.3e}, {ver_max.max() / E0:.3e}]，每周期均值漂移 '
          f'{ver_mean[-1] - ver_mean[0]:.1e}')


# =============================================================================
# 图 4：收敛阶验证
# =============================================================================
def plot_convergence_order(save_path):
    omega, y0, t_final = 1.0, np.array([1.0, 0.0]), 1.0
    dts = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    x_exact = np.cos(omega * t_final)
    f = lambda t, y: harmonic_oscillator_rhs(t, y, omega)
    dH_dq, dH_dp = (lambda q, p: omega ** 2 * q), (lambda q, p: p)
    errs = {'Euler': [], 'SympEuler': [], 'Verlet': [], 'RK4': []}
    for dt in dts:
        _, y = forward_euler(f, y0, (0, t_final), dt)
        errs['Euler'].append(abs(y[-1, 0] - x_exact))
        _, y = rk4(f, y0, (0, t_final), dt)
        errs['RK4'].append(abs(y[-1, 0] - x_exact))
        _, q, _ = symplectic_euler(dH_dq, dH_dp, 1.0, 0.0, (0, t_final), dt)
        errs['SympEuler'].append(abs(q[-1] - x_exact))
        _, q, _ = stormer_verlet(dH_dq, dH_dp, 1.0, 0.0, (0, t_final), dt)
        errs['Verlet'].append(abs(q[-1] - x_exact))

    fig, ax = plt.subplots(figsize=fig_size(1, 1, width=4.2, aspect=0.8))
    spec = [('Euler', C_EULER, 'o', '前向 Euler'), ('SympEuler', C_SYMP, 'd', '辛 Euler'),
            ('Verlet', C_VERLET, '^', 'Verlet'), ('RK4', C_RK4, 's', 'RK4')]
    for key, col, mk, name in spec:
        e = np.array(errs[key])
        slope = np.polyfit(np.log(dts), np.log(e), 1)[0]
        ax.loglog(dts, e, marker=mk, color=col, label=f'{name}：拟合斜率 {slope:.2f}')
        print(f'  {name}: 拟合收敛阶 {slope:.3f}')
    # 参考斜率三角
    for order, col in ((1, C_EULER), (2, C_VERLET), (4, C_RK4)):
        ref = np.array(errs['Euler' if order == 1 else 'Verlet' if order == 2 else 'RK4'])[0] * 3
        ax.loglog(dts, ref * (dts / dts[0]) ** order, color=col, ls=':', lw=0.9)
        ax.text(dts[0] * 1.12, ref, f'$O(\\Delta t^{order})$', color=col, fontsize=8, va='center', ha='left')
    ax.set_xlabel('步长 $\\Delta t$')
    ax.set_ylabel('$t=1$ 处的误差 $|x_h(1)-\\cos 1|$')
    ax.set_xlim(4e-3, 0.5)
    ax.legend(loc='lower right', fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, save_path)


# =============================================================================
if __name__ == '__main__':
    out = 'figures/chap05'
    print('[1] 简谐振子对比');       plot_harmonic_comparison(f'{out}/harmonic_comparison')
    print('[2] 单摆对比');           plot_pendulum_comparison(f'{out}/pendulum_comparison')
    print('[3] 长时间积分');         plot_long_time_comparison(f'{out}/long_time_comparison')
    print('[4] 收敛阶验证');         plot_convergence_order(f'{out}/convergence_order')
    print('完成。')
