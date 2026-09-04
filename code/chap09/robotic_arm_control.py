#!/usr/bin/env python3
"""
第9章扩展阅读：连续控制 —— 二连杆机械臂、PD 控制与 CartPole
=================================================================

从仓库根目录运行::

    python3 code/chap09/robotic_arm_control.py

输出（每张同时有 pdf 与 png；正文引用 pdf）：

    figures/chap09/arm_kinematics          二连杆机械臂：正运动学工作空间 / 逆运动学 / 可操作度椭圆
    figures/chap09/rl_control_comparison   传统控制流程 vs 强化学习流程（概念示意图）
    figures/chap09/pd_control_trajectory   PD 控制器跟踪圆轨迹（RK4 仿真真实数据）
    figures/chap09/cartpole_demo           CartPole 环境示意 + 真实训练得到的学习曲线

CartPole 动力学按 Barto, Sutton & Anderson (1983) 的标准方程自行实现（不依赖 gym）。
学习曲线来自两个真的在跑的算法：
    * 交叉熵方法（CEM）+ 线性策略：直接在 5 个参数上做"进化式"搜索；
    * 表格 Q-Learning：把 4 维连续状态离散成格子，再用第9章的更新公式。
两者都用 5 个随机种子，画均值 ± 标准差。
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import NullFormatter, ScalarFormatter

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS  # noqa: E402

setup_style()


# =============================================================================
# 第 1 部分：二连杆平面机械臂
# =============================================================================
class TwoLinkArm:
    """二连杆平面机械臂（无重力的水平面运动）。"""

    def __init__(self, L1=1.0, L2=0.8, m1=1.0, m2=0.8):
        self.L1, self.L2, self.m1, self.m2 = L1, L2, m1, m2
        self.state = np.zeros(4)  # [θ1, θ2, θ1', θ2']

    def forward_kinematics(self, th1, th2):
        x1, y1 = self.L1 * np.cos(th1), self.L1 * np.sin(th1)
        x2 = x1 + self.L2 * np.cos(th1 + th2)
        y2 = y1 + self.L2 * np.sin(th1 + th2)
        return (x1, y1), (x2, y2)

    def inverse_kinematics(self, x, y):
        """余弦定理解析解，取"肘部朝上"（θ2 ≥ 0）的那一支。"""
        r = np.hypot(x, y)
        if r > self.L1 + self.L2 or r < abs(self.L1 - self.L2):
            return None, None
        c2 = np.clip((r**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2), -1, 1)
        th2 = np.arccos(c2)
        th1 = np.arctan2(y, x) - np.arctan2(self.L2 * np.sin(th2), self.L1 + self.L2 * np.cos(th2))
        return th1, th2

    def jacobian(self, th1, th2):
        s1, c1 = np.sin(th1), np.cos(th1)
        s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
        return np.array([[-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
                         [self.L1 * c1 + self.L2 * c12, self.L2 * c12]])

    def dynamics(self, state, tau):
        """M(θ) θ'' + C(θ, θ') θ' = τ。"""
        th1, th2, d1, d2 = state
        L1, L2, m1, m2 = self.L1, self.L2, self.m1, self.m2
        M11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * np.cos(th2)
        M12 = m2 * L2**2 + m2 * L1 * L2 * np.cos(th2)
        M22 = m2 * L2**2
        M = np.array([[M11, M12], [M12, M22]])
        h = m2 * L1 * L2 * np.sin(th2)
        C = np.array([[-h * d2, -h * (d1 + d2)], [h * d1, 0.0]])
        dd = np.linalg.solve(M, tau - C @ np.array([d1, d2]))
        return np.array([d1, d2, dd[0], dd[1]])

    def step(self, tau, dt=0.01):
        """RK4 积分一步。"""
        f = self.dynamics
        s = self.state
        k1 = f(s, tau)
        k2 = f(s + 0.5 * dt * k1, tau)
        k3 = f(s + 0.5 * dt * k2, tau)
        k4 = f(s + dt * k3, tau)
        self.state = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state.copy()


class PDController:
    """关节空间 PD 控制：τ = Kp (θ_target − θ) − Kd θ'。"""

    def __init__(self, Kp=100.0, Kd=20.0):
        self.Kp, self.Kd = Kp, Kd

    def compute(self, theta, theta_target, dtheta):
        return self.Kp * (theta_target - theta) - self.Kd * dtheta


def plot_arm_kinematics():
    arm = TwoLinkArm()
    fig, axes = plt.subplots(1, 3, figsize=fig_size(3, aspect=1.05))

    # (a) 正运动学：扫描关节角，得到工作空间
    ax = axes[0]
    th1_list = np.radians([-90, -45, 0, 45, 90])
    th2_list = np.radians(np.linspace(-90, 90, 7))
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, len(th1_list)))
    for th1, col in zip(th1_list, shades):
        for th2 in th2_list:
            (x1, y1), (x2, y2) = arm.forward_kinematics(th1, th2)
            ax.plot([0, x1, x2], [0, y1, y2], '-', color=col, lw=1.0, alpha=0.75)
            ax.plot(x2, y2, 'o', color=col, markersize=2.2)
    for th1, col in zip(th1_list, shades):
        ax.plot([], [], '-', color=col, lw=1.5, label=rf'$\theta_1={np.degrees(th1):.0f}^\circ$')
    tt = np.linspace(0, 2 * np.pi, 200)
    for rr in (arm.L1 + arm.L2, abs(arm.L1 - arm.L2)):
        ax.plot(rr * np.cos(tt), rr * np.sin(tt), '--', color=COLORS['red'], lw=1.0)
    ax.plot([], [], '--', color=COLORS['red'], lw=1.0, label='工作空间边界')
    ax.plot(0, 0, 's', color=COLORS['black'], markersize=5)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.0)
    panel_label(ax, '(a)')

    # (b) 逆运动学：到达 4 个目标点
    ax = axes[1]
    targets = [(1.2, 0.5), (0.8, 1.0), (1.5, 0.2), (0.5, 0.8)]
    cols = [COLORS['blue'], COLORS['orange'], COLORS['purple'], COLORS['teal']]
    for (xt, yt), col in zip(targets, cols):
        th1, th2 = arm.inverse_kinematics(xt, yt)
        (x1, y1), (x2, y2) = arm.forward_kinematics(th1, th2)
        ax.plot([0, x1, x2], [0, y1, y2], '-o', color=col, lw=1.8, markersize=3.5,
                label=rf'目标 $({xt},\,{yt})$：$\theta=({np.degrees(th1):.0f}^\circ,{np.degrees(th2):.0f}^\circ)$')
        ax.plot(xt, yt, 's', color=col, markersize=6, markerfacecolor='none', markeredgewidth=1.3)
    ax.plot(0, 0, 's', color=COLORS['black'], markersize=5)
    ax.set_xlim(-0.4, 1.9)
    ax.set_ylim(-0.6, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=1,
              handlelength=1.2, borderaxespad=0.0)
    panel_label(ax, '(b)')

    # (c) 雅可比矩阵的可操作度椭圆：单位关节速度圆 → 末端速度椭圆
    ax = axes[2]
    th1, th2 = np.pi / 4, np.pi / 6
    (x1, y1), (x2, y2) = arm.forward_kinematics(th1, th2)
    J = arm.jacobian(th1, th2)
    U, sv, _ = np.linalg.svd(J)
    scale = 0.35
    circ = np.array([np.cos(tt), np.sin(tt)])
    ell = J @ circ * scale
    ax.plot([0, x1, x2], [0, y1, y2], '-o', color=COLORS['black'], lw=2.2, markersize=4)
    ax.plot(0, 0, 's', color=COLORS['black'], markersize=5)
    ax.fill(x2 + ell[0], y2 + ell[1], color=COLORS['blue'], alpha=0.18, lw=0)
    ax.plot(x2 + ell[0], y2 + ell[1], color=COLORS['blue'], lw=1.5, label=r'$\{J\dot\theta:\ \|\dot\theta\|=1\}$')
    for k in range(2):
        d = U[:, k] * sv[k] * scale
        ax.add_patch(FancyArrowPatch((x2, y2), (x2 + d[0], y2 + d[1]), arrowstyle='-|>',
                                     mutation_scale=9, color=COLORS['red'], lw=1.5, zorder=5))
        ax.annotate(rf'$\sigma_{k + 1}={sv[k]:.2f}$', xy=(x2 + d[0], y2 + d[1]),
                    xytext=(4, 3) if k == 0 else (6, -4), textcoords='offset points',
                    fontsize=7.5, color=COLORS['red'])
    ax.text(0.5 * x1 + 0.02, 0.5 * y1 - 0.12, rf'$\theta_1={np.degrees(th1):.0f}^\circ$', fontsize=7.5)
    ax.text(x1 + 0.05, y1 - 0.02, rf'$\theta_2={np.degrees(th2):.0f}^\circ$', fontsize=7.5, va='top')
    ax.set_xlim(-0.4, 1.9)
    ax.set_ylim(-0.6, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.26), handlelength=1.4,
              borderaxespad=0.0)
    panel_label(ax, '(c)')

    fig.subplots_adjust(wspace=0.38)
    save_figure(fig, 'figures/chap09/arm_kinematics')
    print(f"  可操作度椭圆：σ1 = {sv[0]:.3f}, σ2 = {sv[1]:.3f}, |det J| = {abs(np.linalg.det(J)):.3f}")


# =============================================================================
# 第 2 部分：PD 控制跟踪圆轨迹
# =============================================================================
PD_KP, PD_KD = 400.0, 40.0


def simulate_pd_tracking(Kp=PD_KP, Kd=PD_KD, t_total=5.0, dt=0.01, tau_max=50.0):
    arm = TwoLinkArm()
    ctrl = PDController(Kp, Kd)
    arm.state = np.zeros(4)  # 从伸直姿态 θ = (0, 0) 静止出发
    cx, cy, radius = 1.0, 0.5, 0.3
    t = np.arange(0, t_total, dt)
    log = {k: [] for k in ('t', 'x', 'y', 'xt', 'yt', 'th1', 'th2', 'tau1', 'tau2')}
    for ti in t:
        xt = cx + radius * np.cos(2 * np.pi * ti / t_total)
        yt = cy + radius * np.sin(2 * np.pi * ti / t_total)
        th_t = np.array(arm.inverse_kinematics(xt, yt))
        tau = np.clip(ctrl.compute(arm.state[:2], th_t, arm.state[2:]), -tau_max, tau_max)
        arm.step(tau, dt)
        _, (x, y) = arm.forward_kinematics(arm.state[0], arm.state[1])
        for k, v in zip(log, (ti, x, y, xt, yt, arm.state[0], arm.state[1], tau[0], tau[1])):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


def plot_pd_control_trajectory():
    d = simulate_pd_tracking()
    err = np.hypot(d['x'] - d['xt'], d['y'] - d['yt'])
    print(f"  PD 跟踪（Kp={PD_KP:.0f}, Kd={PD_KD:.0f}）：初始误差 {err[0] * 1e3:.0f} mm，"
          f"1 s 后最大误差 {err[d['t'] > 1].max() * 1e3:.1f} mm，稳态平均误差 {err[d['t'] > 2].mean() * 1e3:.1f} mm，"
          f"力矩饱和的时间比例 {np.mean((np.abs(d['tau1']) >= 50) | (np.abs(d['tau2']) >= 50)):.1%}")

    fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2, aspect=0.66))
    ax = axes[0, 0]
    ax.plot(d['xt'], d['yt'], '--', color=COLORS['black'], lw=1.3, label='目标轨迹（圆）')
    ax.plot(d['x'], d['y'], '-', color=COLORS['blue'], lw=1.4, label='实际末端轨迹')
    ax.plot(d['x'][0], d['y'][0], 'o', color=COLORS['green'], markersize=6, label='起始位置', zorder=5)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_xlim(0.45, 1.95)
    ax.set_ylim(-0.3, 0.95)
    ax.legend(fontsize=7.2, loc='lower left', handlelength=1.6, borderaxespad=0.2)
    panel_label(ax, '(a)')

    ax = axes[0, 1]
    ax.semilogy(d['t'], err * 1e3, color=COLORS['red'], lw=1.4)
    ax.set_yticks([10, 30, 100, 300, 1000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(8, 1200)
    ax.axhline(err[d['t'] > 2].mean() * 1e3, color=COLORS['gray'], ls=':', lw=0.9)
    ax.text(4.95, err[d['t'] > 2].mean() * 1e3 * 1.12, f'稳态约 {err[d["t"] > 2].mean() * 1e3:.0f} mm（跟踪滞后）',
            fontsize=7, color=COLORS['gray'], ha='right', va='bottom')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置误差 (mm)，对数轴')
    panel_label(ax, '(b)')

    ax = axes[1, 0]
    ax.plot(d['t'], np.degrees(d['th1']), color=COLORS['blue'], lw=1.4, label=r'$\theta_1$')
    ax.plot(d['t'], np.degrees(d['th2']), color=COLORS['orange'], lw=1.4, label=r'$\theta_2$')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('关节角 (°)')
    ax.legend(fontsize=8, loc='lower right')
    panel_label(ax, '(c)')

    ax = axes[1, 1]
    ax.plot(d['t'], d['tau1'], color=COLORS['blue'], lw=1.2, label=r'$\tau_1$')
    ax.plot(d['t'], d['tau2'], color=COLORS['orange'], lw=1.2, label=r'$\tau_2$')
    ax.axhline(50, color=COLORS['gray'], ls=':', lw=0.9)
    ax.axhline(-50, color=COLORS['gray'], ls=':', lw=0.9)
    ax.text(4.95, 52, '力矩限幅 ±50', fontsize=7, color=COLORS['gray'], ha='right', va='bottom')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('控制力矩 (N·m)')
    ax.legend(fontsize=8, loc='lower right')
    panel_label(ax, '(d)')

    fig.subplots_adjust(wspace=0.32, hspace=0.45)
    save_figure(fig, 'figures/chap09/pd_control_trajectory')


# =============================================================================
# 第 3 部分：CartPole —— 自行实现的环境 + 两个真实训练的算法
# =============================================================================
class CartPole:
    """Barto, Sutton & Anderson (1983) 的倒立摆小车；与 gym CartPole 相同的参数与欧拉积分。"""
    gravity, masscart, masspole, half_length = 9.8, 1.0, 0.1, 0.5
    force_mag, tau = 10.0, 0.02
    theta_limit, x_limit = 12 * 2 * np.pi / 360, 2.4

    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.half_length

    def reset(self, rng):
        self.state = rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, th, th_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_t, sin_t = np.cos(th), np.sin(th)
        temp = (force + self.polemass_length * th_dot**2 * sin_t) / self.total_mass
        th_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.half_length * (4.0 / 3.0 - self.masspole * cos_t**2 / self.total_mass))
        x_acc = temp - self.polemass_length * th_acc * cos_t / self.total_mass
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        th += self.tau * th_dot
        th_dot += self.tau * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.t += 1
        failed = abs(x) > self.x_limit or abs(th) > self.theta_limit
        done = failed or self.t >= self.max_steps
        return self.state.copy(), 1.0, done


def run_episode(env, policy, rng):
    s = env.reset(rng)
    total = 0.0
    while True:
        s, r, done = env.step(policy(s))
        total += r
        if done:
            return total


def random_policy_baseline(n_seeds=5, n_episodes=200):
    env = CartPole()
    means = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(100 + seed)
        means.append(np.mean([run_episode(env, lambda s: int(rng.integers(2)), rng)
                              for _ in range(n_episodes)]))
    return float(np.mean(means)), float(np.std(means))


def cem_train(seed, n_iter=15, pop=40, elite_frac=0.2):
    """交叉熵方法：线性策略 a = 1[w·s + b > 0]，在 (w, b) 的高斯分布上迭代收紧。"""
    env = CartPole()
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(5), np.ones(5)
    n_elite = max(2, int(pop * elite_frac))
    curve = []  # (累计回合数, 本代种群平均回报, 本代最好回报)
    for it in range(n_iter):
        thetas = mean + std * rng.standard_normal((pop, 5))
        rets = np.array([run_episode(env, lambda s, th=th: int(th[:4] @ s + th[4] > 0), rng)
                         for th in thetas])
        elite = thetas[np.argsort(rets)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 0.02
        curve.append(((it + 1) * pop, rets.mean(), rets.max()))
    return np.array(curve)


def make_discretizer(n_bins=(3, 3, 6, 12)):
    """把 4 维连续状态切成格子；x, ẋ 粗切，θ, θ̇ 细切。"""
    lows = np.array([-2.4, -2.0, -CartPole.theta_limit, -2.0])
    highs = -lows
    edges = [np.linspace(lo, hi, n + 1)[1:-1] for lo, hi, n in zip(lows, highs, n_bins)]

    def discretize(s):
        return tuple(int(np.digitize(v, e)) for v, e in zip(s, edges))
    return discretize, n_bins


def q_learning_cartpole(seed, episodes=2000, gamma=0.99):
    """表格 Q-Learning：ε 与 α 随回合数按对数衰减（经典设置）。"""
    env = CartPole()
    rng = np.random.default_rng(seed)
    discretize, n_bins = make_discretizer()
    Q = np.zeros(tuple(n_bins) + (2,))
    returns = np.zeros(episodes)
    for ep in range(episodes):
        eps = max(0.02, min(1.0, 1.0 - np.log10((ep + 1) / 25)))
        alpha = max(0.1, min(0.5, 1.0 - np.log10((ep + 1) / 25)))
        s = discretize(env.reset(rng))
        total = 0.0
        while True:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2_cont, r, done = env.step(a)
            s2 = discretize(s2_cont)
            failed = done and env.t < env.max_steps
            target = r if failed else r + gamma * Q[s2].max()
            Q[s + (a,)] += alpha * (target - Q[s + (a,)])
            total += r
            s = s2
            if done:
                break
        returns[ep] = total
    return returns


def plot_cartpole_demo():
    t0 = time.time()
    rand_mean, rand_std = random_policy_baseline()
    print(f"  随机策略：平均回报 {rand_mean:.1f} ± {rand_std:.1f}")

    n_seeds = 5
    cem_curves = np.array([cem_train(seed) for seed in range(n_seeds)])  # (seeds, iters, 3)
    cem_ep = cem_curves[0, :, 0]
    cem_mean, cem_sd = cem_curves[:, :, 1].mean(0), cem_curves[:, :, 1].std(0)
    print("  CEM（线性策略，每代 40 回合）：各代种群平均回报 =",
          np.array2string(cem_mean, precision=0, max_line_width=200))
    for th in np.flatnonzero(cem_mean >= 195)[:1]:
        print(f"  CEM 在第 {th + 1} 代（累计 {int(cem_ep[th])} 回合）平均回报首次 ≥ 195")

    EP, W = 2000, 50
    ql_runs = np.array([q_learning_cartpole(seed, EP) for seed in range(n_seeds)])
    ql_ma = np.array([np.convolve(r, np.ones(W) / W, mode='valid') for r in ql_runs])
    ql_mean, ql_sd = ql_ma.mean(0), ql_ma.std(0)
    x_ma = np.arange(W, EP + 1)
    print(f"  表格 Q-Learning：最后 100 回合平均回报 {ql_runs[:, -100:].mean():.1f}"
          f"（各种子：{np.array2string(ql_runs[:, -100:].mean(1), precision=0)}），"
          f"滑动平均峰值 {ql_mean.max():.0f}")
    hit = np.flatnonzero(ql_mean >= 195)
    print("  Q-Learning 滑动平均首次 ≥ 195 的回合：", int(x_ma[hit[0]]) if len(hit) else "未达到")
    print(f"  训练耗时 {time.time() - t0:.0f} s")

    fig, axes = plt.subplots(1, 2, figsize=fig_size(2, aspect=0.74), gridspec_kw={'width_ratios': [1, 1.15]})

    # (a) 环境示意
    ax = axes[0]
    ax.plot([-2.6, 2.6], [0, 0], color=COLORS['black'], lw=2)
    cart_w, cart_h, cart_y = 0.8, 0.4, 0.28
    ax.add_patch(Rectangle((-cart_w / 2, cart_y - cart_h / 2), cart_w, cart_h,
                           fc='#dbe7f3', ec=COLORS['black'], lw=1.2))
    for wx in (-0.25, 0.25):
        ax.add_patch(Circle((wx, 0.08), 0.08, fc=COLORS['gray'], ec=COLORS['black'], lw=0.8))
    th = np.radians(15)
    pole_len, pivot = 1.6, (0.0, cart_y + cart_h / 2)
    tip = (pivot[0] + pole_len * np.sin(th), pivot[1] + pole_len * np.cos(th))
    ax.plot([pivot[0], tip[0]], [pivot[1], tip[1]], color=COLORS['orange'], lw=5, solid_capstyle='round')
    ax.plot(*tip, 'o', color=COLORS['red'], markersize=7)
    ax.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] + 1.2], ls=':', color=COLORS['gray'], lw=1)
    ax.add_patch(Arc(pivot, 1.6, 1.6, angle=0, theta1=90 - np.degrees(th), theta2=90,
                     color=COLORS['red'], lw=1.2))
    ax.text(0.16, pivot[1] + 0.95, r'$\theta$', color=COLORS['red'], fontsize=10)
    ax.add_patch(FancyArrowPatch((cart_w / 2 + 0.05, cart_y), (cart_w / 2 + 0.85, cart_y),
                                 arrowstyle='-|>', mutation_scale=11, color=COLORS['green'], lw=1.8))
    ax.add_patch(FancyArrowPatch((-cart_w / 2 - 0.05, cart_y), (-cart_w / 2 - 0.85, cart_y),
                                 arrowstyle='-|>', mutation_scale=11, color=COLORS['green'], lw=1.8))
    ax.text(cart_w / 2 + 0.45, cart_y + 0.15, '向右推', color=COLORS['green'], fontsize=8, ha='center')
    ax.text(-cart_w / 2 - 0.45, cart_y + 0.15, '向左推', color=COLORS['green'], fontsize=8, ha='center')
    ax.add_patch(FancyArrowPatch((-2.4, -0.32), (0, -0.32), arrowstyle='<->', mutation_scale=8,
                                 color=COLORS['gray'], lw=0.9))
    ax.text(-1.2, -0.42, r'位置 $x$', color=COLORS['gray'], fontsize=8, ha='center', va='top')
    ax.text(0, 3.45, r'$s=(x,\ \dot x,\ \theta,\ \dot\theta)$，$a\in\{$左, 右$\}$', fontsize=8,
            ha='center', va='bottom')
    ax.text(0, 3.02, r'每步 $+1$；$|\theta|>12^\circ$ 或 $|x|>2.4$ 即结束', fontsize=8,
            ha='center', va='bottom')
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-0.9, 3.95)
    ax.set_aspect('equal')
    ax.axis('off')
    panel_label(ax, '(a)', x=0.0, y=0.98)

    # (b) 真实学习曲线
    ax = axes[1]
    ax.plot(x_ma, ql_mean, color=COLORS['blue'], lw=1.5, label='表格 Q-Learning（状态离散化）')
    ax.fill_between(x_ma, ql_mean - ql_sd, ql_mean + ql_sd, color=COLORS['blue'], alpha=0.2, lw=0)
    ax.plot(cem_ep, cem_mean, '-o', color=COLORS['orange'], lw=1.5, markersize=3.5,
            label='交叉熵方法 + 线性策略')
    ax.fill_between(cem_ep, cem_mean - cem_sd, cem_mean + cem_sd, color=COLORS['orange'], alpha=0.2, lw=0)
    ax.axhline(rand_mean, color=COLORS['gray'], ls='--', lw=1.1, label=f'随机策略（{rand_mean:.0f}）')
    ax.axhline(195, color=COLORS['black'], ls=':', lw=1.1, label='"解决"阈值 195')
    ax.set_xlim(0, EP)
    ax.set_ylim(0, 215)
    ax.set_xlabel('训练回合数')
    ax.set_ylabel('回合总奖励（坚持步数）')
    ax.legend(fontsize=7, loc='lower right', handlelength=1.8)
    panel_label(ax, '(b)', x=-0.14)

    fig.subplots_adjust(wspace=0.22)
    save_figure(fig, 'figures/chap09/cartpole_demo')


# =============================================================================
# 第 4 部分：传统控制流程 vs 强化学习流程（概念示意图）
# =============================================================================
def _box(ax, xy, w, h, text, fc='#f4f4f4', ec=None, fontsize=8):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=ec or COLORS['black'], lw=1.0))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, linespacing=1.25)


def _arrow(ax, p, q, color=None, ls='-', rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, color=color or COLORS['gray'],
                                 lw=lw, linestyle=ls, connectionstyle=f'arc3,rad={rad}', zorder=3))


def plot_rl_control_comparison():
    fig, axes = plt.subplots(1, 2, figsize=fig_size(2, aspect=0.8))

    # (a) 传统控制：一条串行流水线，出问题就回头
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    top = [(1.7, 6.0, '系统建模\n（运动方程）'), (5.0, 6.0, '控制器设计\n（如 PD）'),
           (8.3, 6.0, '稳定性分析')]
    bot = [(8.3, 3.2, '参数整定\n$K_p,\\ K_d$'), (5.0, 3.2, '实现与测试'), (1.7, 3.2, '部署')]
    for x, y, t in top + bot:
        _box(ax, (x, y), 2.6, 1.5, t)
    _arrow(ax, (3.0, 6.0), (3.7, 6.0))
    _arrow(ax, (6.3, 6.0), (7.0, 6.0))
    _arrow(ax, (8.3, 5.25), (8.3, 3.95))
    _arrow(ax, (7.0, 3.2), (6.3, 3.2))
    _arrow(ax, (3.7, 3.2), (3.0, 3.2))
    _arrow(ax, (5.0, 2.45), (1.7, 5.25), color=COLORS['red'], ls='--', rad=0.35)
    ax.text(5.0, 1.75, '不满足指标：回头改模型或增益', fontsize=7, color=COLORS['red'], ha='center', va='top')
    ax.text(5.0, 0.3, '需要：专家知识、足够精确的模型', fontsize=8, ha='center', va='bottom',
            style='italic', color=COLORS['gray'])
    panel_label(ax, '(a)', x=0.0, y=0.98)

    # (b) 强化学习：智能体—环境闭环，靠奖励驱动
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    _box(ax, (1.7, 6.0), 2.6, 1.5, '定义奖励\n$r(s,a)$', fc='#f7ecc9')
    _box(ax, (5.0, 6.0), 2.6, 1.5, '智能体\n策略 $\\pi_\\theta(a\\mid s)$', fc='#dbe7f3')
    _box(ax, (8.3, 6.0), 2.6, 1.5, '环境\n（真实或仿真）', fc='#d7ecd9')
    _box(ax, (5.0, 2.6), 4.4, 1.5, '训练循环\n采样轨迹 → 估计回报 → 更新 $\\theta$', fc='#f4f4f4')
    _arrow(ax, (3.0, 6.0), (3.7, 6.0))
    _arrow(ax, (6.3, 6.45), (7.0, 6.45), color=COLORS['blue'], rad=-0.5)
    _arrow(ax, (7.0, 5.55), (6.3, 5.55), color=COLORS['red'], rad=-0.5)
    ax.text(6.65, 7.35, '动作 $a_t$', fontsize=7.5, color=COLORS['blue'], ha='center')
    ax.text(8.3, 4.95, '状态 $s_{t+1}$、奖励 $r_t$', fontsize=7.5, color=COLORS['red'], ha='center', va='top')
    _arrow(ax, (5.6, 5.25), (5.6, 3.35))
    ax.text(5.8, 4.05, '交互数据\n$(s_t,a_t,r_t)$', fontsize=7, color=COLORS['gray'], ha='left', va='center')
    _arrow(ax, (4.4, 3.35), (4.4, 5.25), color=COLORS['blue'], ls='--')
    ax.text(4.2, 4.3, '更新后\n的 $\\theta$', fontsize=7, color=COLORS['blue'], ha='right', va='center')
    ax.text(5.0, 0.3, '需要：奖励设计、大量样本与算力', fontsize=8, ha='center', va='bottom',
            style='italic', color=COLORS['gray'])
    panel_label(ax, '(b)', x=0.0, y=0.98)

    fig.subplots_adjust(wspace=0.08)
    save_figure(fig, 'figures/chap09/rl_control_comparison')


if __name__ == '__main__':
    print("=" * 60)
    print("第9章扩展阅读：连续控制")
    print("=" * 60)
    print("\n[1] 机械臂运动学")
    plot_arm_kinematics()
    print("\n[2] PD 控制跟踪圆轨迹")
    plot_pd_control_trajectory()
    print("\n[3] 传统控制 vs 强化学习（示意图）")
    plot_rl_control_comparison()
    print("\n[4] CartPole：环境 + 真实训练曲线")
    plot_cartpole_demo()
    print("\n全部图已生成。")
