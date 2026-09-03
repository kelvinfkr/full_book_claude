#!/usr/bin/env python3
"""
第16章 图表重绘（v2）
====================

重绘 5 张图（都从仓库根目录运行）::

    cd <仓库根目录>
    python3 code_chap16/chap16_figures_v2.py                 # 全部 5 张
    python3 code_chap16/chap16_figures_v2.py cbf multi       # 只画其中几张

    可选名字：cbf  multi  physics  latent  unified

输出到 figs_chap16/：
    cbf_controller.{pdf,png}         单障碍 CBF 安全滤波器：轨迹 / h(t) / λ*(t)
    multi_obstacle.{pdf,png}         三障碍多约束 CBF-QP：轨迹 / 各 λ_i(t)
    differentiable_physics.{pdf,png} 可微仿真系统辨识（torch autograd 穿过 RK4）
    latent_dynamics.{pdf,png}        单摆图像 → 二维隐空间 → 隐空间多步预测
    unified_view.{pdf,png}           全书统一视角 L = f + λᵀg（概念示意图）

paradigm_comparison / phase_space / free_energy 三张仍由 chap16_world_model.py
生成，本脚本不碰它们。

除 unified_view 是概念示意图外，其余四张图的每条曲线都来自脚本内真实的
求解 / 训练 / 积分；随机种子固定，重跑结果一致。
"""
import itertools
import os
import sys
import time

# 本书的构建环境里多线程 BLAS 反而慢几十倍，统一单线程（须在 import numpy/torch 之前设置）
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, 'code')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch

from textbook_style import COLORS, fig_size, panel_label, save_figure, setup_style

setup_style()
OUT = 'figs_chap16'


# =============================================================================
# 公共：单积分器 ẋ = u 上的 CBF
# =============================================================================
def barrier(x, x_obs, r):
    """圆形障碍的障碍函数 h(x) = |x - x_obs|^2 - r^2 及其梯度 ∇h = 2(x - x_obs)。"""
    d = x - x_obs
    return d @ d - r ** 2, 2.0 * d


def nominal_control(x, goal, v_max=1.0, k_p=10.0):
    """标称控制：朝目标方向走，速度 v_max；只在离目标 v_max/k_p 以内才按比例减速（避免抖振）。"""
    e = goal - x
    dist = np.linalg.norm(e)
    if dist < 1e-12:
        return np.zeros(2)
    return min(v_max, k_p * dist) * e / dist


def cbf_closed_form(x, u_ref, x_obs, r, alpha):
    """单障碍 CBF-QP 的闭式解（正文公式）：
        u* = u_ref + λ* (L_g h)^T,
        λ* = max(0, -(L_f h + L_g h u_ref + α h) / |L_g h|^2)
    单积分器 f=0, g=I，故 L_f h = 0, L_g h = ∇h^T。"""
    h, grad_h = barrier(x, x_obs, r)
    val = grad_h @ u_ref + alpha * h            # 约束左端 L_f h + L_g h u_ref + α h
    lam = max(0.0, -val / (grad_h @ grad_h))
    return u_ref + lam * grad_h, lam, h


def simulate_single(x0, goal, x_obs, r, alpha, use_cbf, dt=0.01, t_max=30.0, tol=0.05):
    """返回 t, x(t), h(t), λ*(t), u_ref(t), u(t)。数组长度一致（最后一点 λ=0）。"""
    x = np.array(x0, float)
    ts, xs, hs, lams, urefs, us = [0.0], [x.copy()], [], [], [], []
    for i in range(int(t_max / dt)):
        u_ref = nominal_control(x, goal)
        if use_cbf:
            u, lam, h = cbf_closed_form(x, u_ref, x_obs, r, alpha)
        else:
            h, _ = barrier(x, x_obs, r)
            u, lam = u_ref, 0.0
        hs.append(h); lams.append(lam); urefs.append(u_ref.copy()); us.append(u.copy())
        x = x + dt * u
        ts.append((i + 1) * dt); xs.append(x.copy())
        if np.linalg.norm(goal - x) < tol:
            break
    h_end, _ = barrier(x, x_obs, r)
    hs.append(h_end); lams.append(0.0)
    urefs.append(nominal_control(x, goal)); us.append(urefs[-1].copy())
    return (np.array(ts), np.array(xs), np.array(hs), np.array(lams),
            np.array(urefs), np.array(us))


# =============================================================================
# 图 1：cbf_controller —— 单障碍安全滤波器
# =============================================================================
def fig_cbf_controller():
    start, goal = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    x_obs, r, alpha = np.array([5.0, 4.0]), 1.5, 1.0
    # 注意：起点、障碍、目标不能共线——共线时 u_ref 与 ∇h 反向，滤波只会减速而不会绕行。

    t_n, x_n, h_n, _, _, _ = simulate_single(start, goal, x_obs, r, alpha, use_cbf=False)
    t_c, x_c, h_c, lam_c, uref_c, u_c = simulate_single(start, goal, x_obs, r, alpha, use_cbf=True)
    active = lam_c > 1e-9
    print(f'[cbf] 无滤波: min h = {h_n.min():.3f}（<0 表示进入禁区）；'
          f'CBF: min h = {h_c.min():.4f}，到达目标用时 {t_c[-1]:.2f} s，'
          f'λ*>0 的时间 {t_c[active].min():.2f}–{t_c[active].max():.2f} s，'
          f'max λ* = {lam_c.max():.3f}')

    fig = plt.figure(figsize=(6.3, 3.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.12, 1.0], hspace=0.55, wspace=0.32,
                  left=0.07, right=0.98, top=0.93, bottom=0.13)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b)

    # ---- (a) 平面轨迹 ----
    ax_a.add_patch(Circle(x_obs, r, facecolor=COLORS['red'], alpha=0.15, edgecolor='none', zorder=1))
    ax_a.add_patch(Circle(x_obs, r, facecolor='none', edgecolor=COLORS['green'], lw=1.6, zorder=2))
    ax_a.plot(*x_obs, 'x', color=COLORS['red'], ms=6, mew=1.4, zorder=3)
    ax_a.text(x_obs[0], x_obs[1] - 0.55, '禁区 $h<0$', ha='center', va='center',
              fontsize=8, color=COLORS['red'])
    ax_a.text(x_obs[0] + r * 0.75 + 0.12, x_obs[1] - r * 0.66, '$h=0$', ha='left', va='center',
              fontsize=8, color=COLORS['green'])
    ax_a.plot(x_n[:, 0], x_n[:, 1], '--', color=COLORS['red'], lw=1.4, label='无滤波 $u_{\\rm ref}$', zorder=4)
    ax_a.plot(x_c[:, 0], x_c[:, 1], '-', color=COLORS['blue'], lw=1.6, label='CBF 滤波 $u^*$', zorder=5)
    ax_a.plot(x_c[active, 0], x_c[active, 1], '-', color=COLORS['orange'], lw=3.2,
              label='其中 $\\lambda^*>0$ 的区段', zorder=6, solid_capstyle='round')
    ax_a.plot(*start, 'o', color=COLORS['black'], ms=6, zorder=7, label='起点')
    ax_a.plot(*goal, '*', color=COLORS['black'], ms=10, zorder=7, label='目标')

    # 在 λ* 最大处画向量分解：u* = u_ref + λ*∇h
    i_max = int(np.argmax(lam_c))
    p = x_c[i_max]
    _, grad = barrier(p, x_obs, r)
    v_ref, v_corr, v_star = uref_c[i_max], lam_c[i_max] * grad, u_c[i_max]
    scale = 2.2
    unit = lambda v: v / np.linalg.norm(v)
    cw = lambda v: np.array([v[1], -v[0]]) / np.linalg.norm(v)      # 顺时针垂直单位向量
    arrow_kw = dict(arrowstyle='-|>', mutation_scale=9, lw=1.3, shrinkA=0, shrinkB=0)
    tip_ref, tip_star = p + scale * v_ref, p + scale * v_star
    ax_a.annotate('', xy=tip_ref, xytext=p, arrowprops=dict(color=COLORS['gray'], **arrow_kw), zorder=8)
    ax_a.annotate('', xy=tip_star, xytext=tip_ref, arrowprops=dict(color=COLORS['green'], **arrow_kw), zorder=8)
    ax_a.annotate('', xy=tip_star, xytext=p, arrowprops=dict(color=COLORS['blue'], **arrow_kw), zorder=8)
    ax_a.plot(*p, 'o', color=COLORS['black'], ms=3.2, zorder=9)
    ax_a.text(*(p + 0.5 * scale * v_ref + 0.42 * cw(v_ref)), '$u_{\\rm ref}$', fontsize=8,
              color=COLORS['gray'], ha='center', va='center')
    ax_a.text(*(tip_ref + 0.5 * scale * v_corr + 0.36 * cw(v_corr)), '$\\lambda^*\\nabla h$',
              fontsize=8, color=COLORS['green'], ha='center', va='center')
    ax_a.text(*(p + 0.55 * scale * v_star - 0.38 * cw(v_star)), '$u^*$', fontsize=8,
              color=COLORS['blue'], ha='center', va='center')

    ax_a.set_xlim(-0.6, 10.6); ax_a.set_ylim(-0.6, 10.6)
    ax_a.set_aspect('equal')
    ax_a.set_xlabel('$x_1$'); ax_a.set_ylabel('$x_2$')
    ax_a.legend(loc='upper left', fontsize=7.5, handlelength=2.2, borderpad=0.4, labelspacing=0.35)
    panel_label(ax_a, '(a)')

    # ---- (b) h(t) ----
    ax_b.axhline(0, color=COLORS['black'], ls='--', lw=1.0)
    ax_b.fill_between([0, max(t_n[-1], t_c[-1])], -12, 0, color=COLORS['red'], alpha=0.10, lw=0)
    ax_b.plot(t_n, h_n, '--', color=COLORS['red'], label='无滤波')
    ax_b.plot(t_c, h_c, '-', color=COLORS['blue'], label='CBF 滤波')
    ax_b.set_ylim(-4, 30)
    ax_b.set_ylabel('$h(x(t))$')
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.52, 1.02), fontsize=7.5, ncol=1,
                borderpad=0.2, labelspacing=0.25)
    ax_b.text(0.98, 0.04, '$h<0$：禁区', transform=ax_b.transAxes, fontsize=7.5,
              color=COLORS['red'], va='bottom', ha='right')
    plt.setp(ax_b.get_xticklabels(), visible=False)
    panel_label(ax_b, '(b)', x=-0.16)

    # ---- (c) λ*(t) ----
    ax_c.fill_between(t_c, 0, lam_c, color=COLORS['orange'], alpha=0.35, lw=0)
    ax_c.plot(t_c, lam_c, '-', color=COLORS['blue'])
    ax_c.set_xlabel('时间 $t$ / s')
    ax_c.set_ylabel('$\\lambda^*(t)$')
    ax_c.set_xlim(0, max(t_n[-1], t_c[-1]))
    ax_c.set_ylim(0, lam_c.max() * 1.25)
    ax_c.annotate('只有约束顶到边界时\n$\\lambda^*$ 才非零',
                  xy=(t_c[i_max], lam_c[i_max]), xytext=(0.62, 0.72), textcoords='axes fraction',
                  fontsize=7.5, ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.8))
    panel_label(ax_c, '(c)', x=-0.16)

    save_figure(fig, f'{OUT}/cbf_controller')


# =============================================================================
# 图 2：multi_obstacle —— 多约束 CBF-QP
# =============================================================================
def cbf_qp_multi(u_ref, x, obstacles, alpha):
    """多约束 CBF-QP：min ½|u-u_ref|²  s.t.  A u + b ≥ 0，A_i = ∇h_i^T，b_i = α h_i。
    KKT：u = u_ref + A^T λ，λ ≥ 0，λ_i (A_i u + b_i) = 0。
    问题只有 2–3 个约束，直接枚举活跃集求精确解（严格凸 QP 的 KKT 点唯一）。"""
    A = np.array([barrier(x, c, r)[1] for c, r in obstacles])
    hs = np.array([barrier(x, c, r)[0] for c, r in obstacles])
    b = alpha * hs
    m = len(b)
    for size in range(m + 1):
        for S in itertools.combinations(range(m), size):
            S = list(S)
            lam = np.zeros(m)
            if S:
                AS = A[S]
                M = AS @ AS.T
                if np.linalg.cond(M) > 1e10:
                    continue
                lam_S = np.linalg.solve(M, -(AS @ u_ref + b[S]))
                if np.any(lam_S < -1e-10):
                    continue
                lam[S] = np.maximum(lam_S, 0.0)
            u = u_ref + A.T @ lam
            if np.all(A @ u + b >= -1e-8):
                return u, lam, hs
    # 退化情形（几乎不会发生）：对偶投影梯度
    Q, c = A @ A.T, A @ u_ref + b
    lam = np.zeros(m); step = 1.0 / (np.linalg.norm(Q, 2) + 1e-12)
    for _ in range(20000):
        lam = np.maximum(lam - step * (Q @ lam + c), 0.0)
    return u_ref + A.T @ lam, lam, hs


def fig_multi_obstacle():
    start, goal = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    # 障碍 1、2 夹出一条正对直线路径的窄缝（缝宽约 0.6），障碍 3 直接挡在直线上
    obstacles = [(np.array([3.1, 5.0]), 1.2),
                 (np.array([5.0, 3.1]), 1.1),
                 (np.array([7.6, 7.9]), 1.3)]
    alpha, dt = 1.0, 0.01
    obs_colors = [COLORS['orange'], COLORS['purple'], COLORS['teal']]

    x = start.copy()
    ts, xs, lams, hs = [0.0], [x.copy()], [], []
    for i in range(int(40.0 / dt)):
        u_ref = nominal_control(x, goal)
        u, lam, h = cbf_qp_multi(u_ref, x, obstacles, alpha)
        lams.append(lam); hs.append(h)
        x = x + dt * u
        ts.append((i + 1) * dt); xs.append(x.copy())
        if np.linalg.norm(goal - x) < 0.05:
            break
    lams.append(np.zeros(len(obstacles)))
    hs.append(np.array([barrier(x, c, r)[0] for c, r in obstacles]))
    ts, xs, lams, hs = map(np.array, (ts, xs, lams, hs))
    print(f'[multi] 到达用时 {ts[-1]:.2f} s，各约束 min h = {hs.min(0).round(4)}，'
          f'max λ_i = {lams.max(0).round(3)}，'
          f'同时两个约束活跃的步数 = {int(((lams > 1e-9).sum(1) >= 2).sum())}')
    # 直线参考路径（无滤波）
    x_line = np.linspace(start, goal, 200)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 3.05),
                                     gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.35,
                                                      left=0.07, right=0.98, top=0.92, bottom=0.15))
    # ---- (a) ----
    for k, ((c, r), col) in enumerate(zip(obstacles, obs_colors)):
        ax_a.add_patch(Circle(c, r, facecolor=col, alpha=0.22, edgecolor=col, lw=1.4, zorder=1))
        ax_a.text(c[0], c[1], f'{k + 1}', ha='center', va='center', fontsize=9, color=col,
                  fontweight='bold', zorder=2)
    ax_a.plot(x_line[:, 0], x_line[:, 1], '--', color=COLORS['red'], lw=1.2, label='无滤波（直线）', zorder=3)
    ax_a.plot(xs[:, 0], xs[:, 1], '-', color=COLORS['blue'], lw=1.6, label='CBF-QP 滤波', zorder=4)
    # 把活跃约束对应的区段用障碍颜色标出；先画的点更大，两个约束同时活跃时呈"光环 + 内核"
    for k, col in enumerate(obs_colors):
        act = lams[:, k] > 1e-9
        if act.any():
            ax_a.plot(xs[act, 0], xs[act, 1], '.', color=col, ms=6.0 - 1.4 * k, zorder=5 + k,
                      label=f'约束 {k + 1} 活跃（$\\lambda_{k + 1}>0$）')
    ax_a.plot(*start, 'o', color=COLORS['black'], ms=6, zorder=9)
    ax_a.plot(*goal, '*', color=COLORS['black'], ms=10, zorder=9)
    ax_a.text(start[0] + 0.3, start[1] - 0.2, '起点', fontsize=8, va='top')
    ax_a.text(goal[0] - 0.3, goal[1] + 0.1, '目标', fontsize=8, ha='right', va='bottom')
    ax_a.set_xlim(-0.6, 10.6); ax_a.set_ylim(-0.6, 10.6); ax_a.set_aspect('equal')
    ax_a.set_xlabel('$x_1$'); ax_a.set_ylabel('$x_2$')
    ax_a.legend(loc='upper left', fontsize=7, handlelength=1.8, borderpad=0.3, labelspacing=0.3)
    panel_label(ax_a, '(a)')

    # ---- (b) λ_i(t) ----
    for k, col in enumerate(obs_colors):
        ax_b.plot(ts, lams[:, k], '-', color=col, label=f'$\\lambda_{k + 1}$（障碍 {k + 1}）')
    both = (lams > 1e-9).sum(1) >= 2
    if both.any():
        t0, t1 = ts[both].min(), ts[both].max()
        ax_b.axvspan(t0, t1, color=COLORS['gray'], alpha=0.15, lw=0)
        ax_b.annotate('两个约束同时活跃\n（穿过 1、2 之间的缝隙）', xy=(0.5 * (t0 + t1), lams[both].max() * 1.05),
                      xytext=(0.04, 0.62), textcoords='axes fraction', fontsize=7.5, ha='left',
                      arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.8))
    ax_b.set_xlabel('时间 $t$ / s'); ax_b.set_ylabel('$\\lambda_i(t)$')
    ax_b.set_xlim(0, ts[-1]); ax_b.set_ylim(0, lams.max() * 1.3)
    ax_b.legend(loc='upper right', fontsize=7.5)
    panel_label(ax_b, '(b)', x=-0.13)
    save_figure(fig, f'{OUT}/multi_obstacle')


# =============================================================================
# 图 3：differentiable_physics —— 可微仿真做系统辨识
# =============================================================================
def fig_differentiable_physics():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m_true, k_true, c_true = 1.0, 10.0, 0.5
    p_init = np.array([2.0, 5.0, 1.0])          # 故意猜错的初值 (m, k, c)
    dt, N = 0.02, 200                            # 4 s，RK4
    t = np.arange(N + 1) * dt
    u_np = 3.0 * np.sin(2.0 * t) + 2.0 * np.sin(5.5 * t)    # 已知外力 u(t)
    u = torch.tensor(u_np, dtype=torch.float64)
    x0, v0 = 1.0, 0.0
    sigma_obs = 0.03

    def simulate(logp):
        """RK4 显式积分 m x'' = -k x - c x' + u；整段是一张可微计算图。"""
        m, k, c = torch.exp(logp[0]), torch.exp(logp[1]), torch.exp(logp[2])
        acc = lambda x, v, ut: (-k * x - c * v + ut) / m
        x = torch.tensor(x0, dtype=torch.float64)
        v = torch.tensor(v0, dtype=torch.float64)
        xs = [x]
        for i in range(N):
            u0, u1 = u[i], u[i + 1]; um = 0.5 * (u0 + u1)
            k1x, k1v = v, acc(x, v, u0)
            k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, um)
            k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, um)
            k4x, k4v = v + dt * k3v, acc(x + dt * k3x, v + dt * k3v, u1)
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            xs.append(x)
        return torch.stack(xs)

    with torch.no_grad():
        x_true = simulate(torch.log(torch.tensor([m_true, k_true, c_true], dtype=torch.float64))).numpy()
    x_obs = x_true + sigma_obs * rng.standard_normal(x_true.shape)
    x_obs_t = torch.tensor(x_obs)

    logp = torch.tensor(np.log(p_init), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logp], lr=0.05)
    n_iter = 300
    hist_loss, hist_p = [], [p_init.copy()]
    with torch.no_grad():
        x_init = simulate(logp).numpy()
    t0 = time.time()
    for it in range(n_iter):
        opt.zero_grad()
        loss = ((simulate(logp) - x_obs_t) ** 2).mean()
        loss.backward()                          # 梯度沿 200 步 RK4 反向传播（伴随）
        opt.step()
        hist_loss.append(loss.item()); hist_p.append(np.exp(logp.detach().numpy()))
    hist_loss = np.array(hist_loss); hist_p = np.array(hist_p)
    with torch.no_grad():
        x_fit = simulate(logp).numpy()
        loss_final = ((x_fit - x_obs) ** 2).mean()
    m_f, k_f, c_f = hist_p[-1]
    print(f'[physics] 用时 {time.time() - t0:.1f} s；损失 {hist_loss[0]:.3e} → {loss_final:.3e}'
          f'（噪声下限 σ² = {sigma_obs ** 2:.1e}）；'
          f'辨识结果 m={m_f:.3f} k={k_f:.3f} c={c_f:.3f}（真值 {m_true}, {k_true}, {c_true}）；'
          f'第 100 次迭代时 m={hist_p[100][0]:.3f} k={hist_p[100][1]:.3f} c={hist_p[100][2]:.3f}')

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35),
                             gridspec_kw=dict(wspace=0.42, left=0.08, right=0.99, top=0.9, bottom=0.2))
    ax_a, ax_b, ax_c = axes
    # ---- (a) 轨迹 ----
    ax_a.plot(t[::4], x_obs[::4], 'o', color=COLORS['gray'], ms=2.2, alpha=0.7, label='带噪观测', zorder=2)
    ax_a.plot(t, x_true, '-', color=COLORS['black'], lw=1.0, label='真实', zorder=3)
    ax_a.plot(t, x_init, '--', color=COLORS['orange'], lw=1.4, label='初始猜测', zorder=4)
    ax_a.plot(t, x_fit, '-', color=COLORS['blue'], lw=1.6, label='辨识后', zorder=5)
    ax_a.set_xlabel('时间 $t$ / s'); ax_a.set_ylabel('位置 $x(t)$')
    ax_a.legend(loc='upper right', fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.6,
                borderpad=0.2, labelspacing=0.25)
    ax_a.set_ylim(x_init.min() * 1.15 - 0.3, x_init.max() * 1.15 + 1.2)
    panel_label(ax_a, '(a)', x=-0.2)
    # ---- (b) 损失 ----
    ax_b.semilogy(np.arange(1, n_iter + 1), hist_loss, '-', color=COLORS['blue'])
    ax_b.axhline(sigma_obs ** 2, color=COLORS['gray'], ls=':', lw=1.0)
    ax_b.text(n_iter * 0.98, sigma_obs ** 2 * 1.35, '观测噪声方差 $\\sigma^2$', ha='right', va='bottom',
              fontsize=7.5, color=COLORS['gray'])
    ax_b.set_xlabel('迭代次数'); ax_b.set_ylabel('均方误差')
    panel_label(ax_b, '(b)', x=-0.2)
    # ---- (c) 参数 ----
    its = np.arange(n_iter + 1)
    for j, (name, tv, col) in enumerate([('$m$', m_true, COLORS['blue']),
                                          ('$k$', k_true, COLORS['orange']),
                                          ('$c$', c_true, COLORS['purple'])]):
        ax_c.semilogy(its, hist_p[:, j], '-', color=col, label=f'{name}（真值 {tv:g}）')
        ax_c.axhline(tv, color=col, ls='--', lw=0.9, alpha=0.7)
    ax_c.set_xlabel('迭代次数'); ax_c.set_ylabel('参数值（对数轴）')
    ax_c.set_ylim(0.3, 20)
    ax_c.legend(loc='center right', fontsize=7.5, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    panel_label(ax_c, '(c)', x=-0.2)
    save_figure(fig, f'{OUT}/differentiable_physics')


# =============================================================================
# 图 4：latent_dynamics —— 单摆图像 → 隐空间 → 隐空间动力学
# =============================================================================
def pendulum_rollout(theta0, omega0, T, dt):
    """无阻尼单摆 θ'' = -sin θ（g/l = 1），RK4，向量化。返回 (T+1, n) 的 θ, ω。"""
    f = lambda th, om: (om, -np.sin(th))
    th, om = theta0.astype(float).copy(), omega0.astype(float).copy()
    TH, OM = [th.copy()], [om.copy()]
    for _ in range(T):
        k1t, k1o = f(th, om)
        k2t, k2o = f(th + 0.5 * dt * k1t, om + 0.5 * dt * k1o)
        k3t, k3o = f(th + 0.5 * dt * k2t, om + 0.5 * dt * k2o)
        k4t, k4o = f(th + dt * k3t, om + dt * k3o)
        th = th + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om = om + dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
        TH.append(th.copy()); OM.append(om.copy())
    return np.array(TH), np.array(OM)


def render_pendulum(theta, S=24, L=8.5, sigma=2.4):
    """把角度渲染成 S×S 灰度图：支点在中心，摆杆（淡）+ 摆锤（亮高斯斑）。theta: (...,)。"""
    theta = np.asarray(theta)
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    c = (S - 1) / 2
    img = np.zeros(theta.shape + (S, S))
    # 斑点取得较大（σ≈2.4 px）：两帧斑点有重叠，像素均方误差对位置才有梯度，编码器才学得动
    for frac, amp, sg in [(0.3, 0.3, 1.2), (0.55, 0.3, 1.2), (0.8, 0.3, 1.2), (1.0, 1.0, sigma)]:
        bx = c + frac * L * np.sin(theta)
        by = c + frac * L * np.cos(theta)      # 图像行向下为正，摆锤挂在支点下方
        img += amp * np.exp(-((xx - bx[..., None, None]) ** 2 + (yy - by[..., None, None]) ** 2) / (2 * sg ** 2))
    return np.clip(img, 0, 1)


def latent_experiment():
    """单摆图像 → 学一个一维"广义坐标" q → 二阶隐空间动力学。返回画图所需的全部数组。

    结构（对应正文的 Encoder / Dynamics / Decoder 三件套）：
        感知  q_t = e(I_t)                       单帧 → 一维坐标（学出来的 θ）
              z_t = (q_t, v_t),  v_t = q_t - q_{t-1}   观测 o_t=[I_t, I_{t-1}] 的二维潜态
        预测  v_{t+1} = v_t + a(q_t, v_t),  q_{t+1} = q_t + v_{t+1}   （隐空间里的辛欧拉，a 是 MLP）
        想象  Î_t = d(q_t)
    损失 = 单帧重建 + 隐空间多步展开后的重建 + 隐空间一致性 + "方差不坍缩"（VICReg 式，正文第1章回顾框）。
    设置环境变量 CHAP16_CACHE=<目录> 可把结果缓存成 .npz（仅用于反复调整版式时省时间）。"""
    cache_dir = os.environ.get('CHAP16_CACHE')
    cache = os.path.join(cache_dir, 'latent_experiment.npz') if cache_dir else None
    if cache and os.path.exists(cache):
        print(f'[latent] 从缓存读取 {cache}')
        return dict(np.load(cache, allow_pickle=True))

    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    S, dt, T = 24, 0.2, 60           # dt=0.2 s：相邻两帧摆锤位移约 2–3 px，速度信息才看得见
    n_train, n_test = 300, 40
    noise = 0.03

    def sample_ic(n):
        th, om = [], []
        while len(th) < n:
            a, b = rng.uniform(-2.6, 2.6), rng.uniform(-1.6, 1.6)
            if 0.5 * b ** 2 + 1 - np.cos(a) < 1.9:      # 只取摆动（不翻越）的能量
                th.append(a); om.append(b)
        return np.array(th), np.array(om)

    def make_dataset(n):
        th0, om0 = sample_ic(n)
        TH, OM = pendulum_rollout(th0, om0, T, dt)      # (T+1, n)
        frames = render_pendulum(TH.T).reshape(n, T + 1, S * S)          # (n, T+1, S²)
        noisy = frames + noise * rng.standard_normal(frames.shape)
        return dict(theta=TH.T, omega=OM.T, clean=frames.astype(np.float32), noisy=noisy.astype(np.float32))

    tr, te = make_dataset(n_train), make_dataset(n_test)
    D = S * S
    print(f'[latent] 单帧 {S}×{S}={D} 像素，观测 o_t=[I_t, I_(t-1)] 共 {2 * D} 维；'
          f'训练序列 {n_train}×{T + 1} 帧，噪声 σ={noise}')

    enc = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    dec = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, D))
    acc = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def dyn(z):
        """z=(q,v) → 下一步：v' = v + a(q,v)，q' = q + v'。"""
        q, v = z[..., :1], z[..., 1:]
        v_new = v + acc(z)
        return torch.cat([q + v_new, v_new], -1)

    params = list(enc.parameters()) + list(dec.parameters()) + list(acc.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    n_steps, B, K = 5000, 48, 12
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=5e-5)
    frames_tr = torch.tensor(tr['noisy'])
    t0 = time.time()
    for step in range(n_steps):
        idx = rng.integers(0, n_train, B); s = rng.integers(1, T + 1 - K, B)
        I = frames_tr[idx[:, None], s[:, None] + np.arange(-1, K + 1)]    # (B, K+2, D)：帧 t-1 … t+K
        q = enc(I)                                                        # (B, K+2, 1)
        z = torch.cat([q[:, 1:], q[:, 1:] - q[:, :-1]], -1)               # 编码得到的 z_t … z_{t+K}
        zhat = [z[:, 0]]
        for _ in range(K):
            zhat.append(dyn(zhat[-1]))
        zhat = torch.stack(zhat, 1)                                       # 隐空间展开 ẑ_t … ẑ_{t+K}
        loss_ae = ((dec(q) - I) ** 2).sum(-1).mean()                      # 感知+想象：单帧重建
        loss_pred = ((dec(zhat[:, 1:, :1]) - I[:, 2:]) ** 2).sum(-1).mean()   # 预测：展开后重建
        loss_lat = ((zhat[:, 1:] - z[:, 1:]) ** 2).sum(-1).mean()         # 隐空间一致性
        loss_var = torch.relu(1.0 - q.std()) ** 2                         # 不许坍缩：q 的标准差不低于 1
        loss_smooth = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]).pow(2).mean()  # 隐坐标沿时间光滑（折叠会被罚）
        loss = loss_ae + loss_pred + 10.0 * loss_lat + 10.0 * loss_var + 10.0 * loss_smooth
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == n_steps - 1:
            print(f'    step {step:5d}  loss {loss.item():.3f}  (ae {loss_ae.item():.3f}, '
                  f'pred {loss_pred.item():.3f}, lat {loss_lat.item():.4f}, std q {q.std().item():.2f})'
                  f'  {time.time() - t0:.0f}s', flush=True)
    train_time = time.time() - t0

    # ---------- 评估（测试集） ----------
    enc.eval(); dec.eval(); acc.eval()
    with torch.no_grad():
        I_te = torch.tensor(te['noisy'])
        q_te = enc(I_te)                                                  # (n_test, T+1, 1)
        z_te_t = torch.cat([q_te[:, 1:], q_te[:, 1:] - q_te[:, :-1]], -1) # z_1 … z_T
        z_te = z_te_t.numpy()
        theta, omega = te['theta'][:, 1:], te['omega'][:, 1:]
        H = 30
        starts = [0, 8, 16, 24]
        err_model = np.zeros((len(starts), n_test, H + 1)); err_hold = np.zeros_like(err_model)
        for si, s0 in enumerate(starts):
            z = z_te_t[:, s0]
            for k in range(H + 1):
                pred = dec(z[:, :1]).numpy()
                truth = te['clean'][:, 1 + s0 + k]
                err_model[si, :, k] = np.sqrt(((pred - truth) ** 2).mean(-1))
                err_hold[si, :, k] = np.sqrt(((te['clean'][:, 1 + s0] - truth) ** 2).mean(-1))
                z = dyn(z)
        rmse_model = err_model.mean((0, 1)); rmse_hold = err_hold.mean((0, 1))
        # 隐坐标与真实状态的光滑对应：三次多项式回归 (θ, ω) 的 R²
        Z = z_te.reshape(-1, 2)
        feats = np.stack([Z[:, 0] ** i * Z[:, 1] ** j for i in range(4) for j in range(4 - i)], 1)
        target = np.stack([theta.ravel(), omega.ravel()], 1)
        coef, *_ = np.linalg.lstsq(feats, target, rcond=None)
        r2 = 1 - (target - feats @ coef).var(0) / target.var(0)
        # 三条不同能量的测试轨迹：编码轨迹 + 从 z_0 出发的纯隐空间展开
        E = 0.5 * omega[:, 0] ** 2 + 1 - np.cos(theta[:, 0])
        order = np.argsort(E)
        picks = np.array([order[2], order[n_test // 2], order[-3]])
        roll = []
        for i in picks:
            z = z_te_t[i, 0:1]; zs = [z]
            for _ in range(T - 1):
                z = dyn(z); zs.append(z)
            roll.append(torch.cat(zs).numpy())
    print(f'[latent] 训练 {train_time:.0f} s；多步预测像素 RMSE：1 步 {rmse_model[1]:.4f}，'
          f'10 步 {rmse_model[10]:.4f}，{H} 步 {rmse_model[H]:.4f}；保持当前帧 {H} 步 {rmse_hold[H]:.4f}；'
          f'噪声 {noise}；隐变量→(θ, ω) 三次多项式回归 R² = {r2.round(3)}')
    out = dict(S=S, dt=dt, noise=noise, H=H, rmse_model=rmse_model, rmse_hold=rmse_hold, r2=r2,
               picks=picks, roll=np.array(roll), z_te=z_te, theta=theta, omega=omega,
               noisy=te['noisy'][:, 1:], train_time=train_time)
    if cache:
        os.makedirs(cache_dir, exist_ok=True); np.savez(cache, **out)
    return out


def fig_latent_dynamics():
    R = latent_experiment()
    S, dt, H, noise = int(R['S']), float(R['dt']), int(R['H']), float(R['noise'])
    picks, roll, z_te = R['picks'], R['roll'], R['z_te']
    theta, omega, noisy = R['theta'], R['omega'], R['noisy']
    rmse_model, rmse_hold = R['rmse_model'], R['rmse_hold']

    fig = plt.figure(figsize=(6.3, 4.1))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.62, 1.55], hspace=0.62, wspace=0.42,
                  left=0.075, right=0.985, top=0.95, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c, ax_d = (fig.add_subplot(gs[1, j]) for j in range(3))
    ep_colors = [COLORS['blue'], COLORS['orange'], COLORS['purple']]

    # (a) 胶片条：某条测试轨迹的 8 帧带噪观测（只显示堆叠中的当前帧）
    i_show = picks[1]
    frame_ids = np.arange(0, 8 * 3, 3)
    strip = np.concatenate([noisy[i_show, k].reshape(S, S) for k in frame_ids], axis=1)
    ax_a.imshow(strip, cmap='gray_r', vmin=-0.1, vmax=1.05, interpolation='nearest', aspect='equal')
    for j in range(1, len(frame_ids)):
        ax_a.axvline(j * S - 0.5, color='white', lw=1.2)
    ax_a.set_xticks(np.arange(len(frame_ids)) * S + S / 2 - 0.5)
    ax_a.set_xticklabels([f'$t={k * dt:.1f}$ s' for k in frame_ids], fontsize=7.5)
    ax_a.set_yticks([]); ax_a.grid(False)
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    ax_a.set_ylabel(f'{S}×{S} 像素', fontsize=8)
    panel_label(ax_a, '(a)', x=-0.03)

    # (b) 真实相空间
    for i, col in zip(picks, ep_colors):
        ax_b.plot(theta[i], omega[i], '-', color=col, lw=1.4)
        ax_b.plot(theta[i, 0], omega[i, 0], 'o', color=col, ms=4)
    ax_b.set_xlabel('角度 $\\theta$ / rad'); ax_b.set_ylabel('角速度 $\\omega$ / rad·s$^{-1}$')
    ax_b.set_title('真实相空间（不可见）', fontsize=8.5, pad=4)
    panel_label(ax_b, '(b)', x=-0.28)

    # (c) 学到的隐空间
    for i, zr, col in zip(picks, roll, ep_colors):
        ax_c.plot(z_te[i, :, 0], z_te[i, :, 1], '-', color=col, lw=1.4)
        ax_c.plot(zr[:, 0], zr[:, 1], ':', color=col, lw=1.4)
        ax_c.plot(z_te[i, 0, 0], z_te[i, 0, 1], 'o', color=col, ms=4)
    ax_c.plot([], [], '-', color=COLORS['gray'], label='编码 $z_t=E(o_t)$')
    ax_c.plot([], [], ':', color=COLORS['gray'], label='隐空间展开 $\\hat z_{t+1}=F(\\hat z_t)$')
    ax_c.set_xlabel('隐坐标 $q_t$'); ax_c.set_ylabel('隐速度 $v_t=q_t-q_{t-1}$')
    ax_c.set_title('学到的二维隐空间', fontsize=8.5, pad=4)
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=7, ncol=1,
                handlelength=1.8, borderpad=0.2, labelspacing=0.2)
    panel_label(ax_c, '(c)', x=-0.28)

    # (d) 多步预测误差
    ks = np.arange(H + 1)
    ax_d.plot(ks, rmse_hold, '--', color=COLORS['gray'], label='保持当前帧不变')
    ax_d.plot(ks, rmse_model, '-', color=COLORS['blue'], label='隐空间展开后解码')
    ax_d.axhline(noise, color=COLORS['red'], ls=':', lw=1.0)
    ax_d.text(H * 0.98, noise * 0.9, '观测噪声水平', ha='right', va='top', fontsize=7, color=COLORS['red'])
    ax_d.set_xlabel('预测步数 $k$'); ax_d.set_ylabel('像素均方根误差')
    ax_d.set_xlim(0, H); ax_d.set_ylim(0, max(rmse_hold.max(), rmse_model.max()) * 1.15)
    ax_d.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=7, ncol=1,
                handlelength=1.8, borderpad=0.2, labelspacing=0.2)
    panel_label(ax_d, '(d)', x=-0.28)

    save_figure(fig, f'{OUT}/latent_dynamics')


# =============================================================================
# 图 5：unified_view —— 全书统一视角（概念示意图）
# =============================================================================
def fig_unified_view():
    """概念示意图：中心 L = f + λᵀg，两侧 8 个章节框写出各自的 f / g / λ。全部用英寸坐标排版。"""
    W, Hh = 6.3, 6.6
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis('off')

    left = [
        ('第1章  约束优化', '任意目标 $f(x)$', '$g(x)=0$，$h(x)\\leq 0$', '影子价格', 'KKT 条件'),
        ('第3章  变分法', '泛函 $\\int F\\,dx$', '$y_{k+1}-y_k=v_k\\Delta x$', '广义动量 $\\partial F/\\partial y\'$', '欧拉–拉格朗日方程'),
        ('第6–7章  机器人 / 最优控制', '代价 $\\int \\ell\\,dt$、能耗', '$\\dot x=f(x,u)$，接触 $\\phi\\geq 0$', '协态 $\\lambda(t)$、接触力', '伴随方程、PMP'),
        ('第16章  具身智能', '$\\frac{1}{2}\\|u-u_{\\rm ref}\\|^2$（贴近标称）', '安全 $h(x)\\geq 0$（CBF）', '安全约束的价格 $\\lambda^*$', 'CBF-QP'),
    ]
    right = [
        ('第8章  神经网络', '损失 $\\ell(z_n)$', '逐层 $z_i=f_i(z_{i-1})$', '误差信号 $\\delta_i$', '反向传播'),
        ('第9章  强化学习', '累积奖励', '$s_{t+1}=f(s_t,a_t)$', '值函数 $V(s)$', '贝尔曼方程'),
        ('第14章  PINN', '数据拟合', 'PDE 残差 $=0$', '边界条件的影子价格（对偶上升）', '物理信息损失'),
        ('第10章  多智能体博弈', '每个玩家各自的代价', '共享资源 / 流量守恒', '均衡价格、拥堵费', '联立 KKT = 纳什均衡'),
    ]
    bw, bh, gap = 1.92, 1.28, 0.16          # 侧框尺寸（英寸）
    cb_w, cb_h = 1.42, 1.5                  # 中心框尺寸
    x_left, x_right = 0.08, W - 0.08 - bw
    cx = W / 2
    y_top = Hh - 0.55
    y_boxes = [y_top - bh - i * (bh + gap) for i in range(4)]
    cb_y0 = (y_boxes[1] + y_boxes[2] + bh) / 2 - cb_h / 2   # 中心框垂直居中于四个侧框
    fs = 7.2

    def draw_box(x0, y0, title, f, g, lam, cond):
        ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    facecolor='#F5F7FA', edgecolor=COLORS['blue'], lw=1.0))
        ax.text(x0 + bw / 2, y0 + bh - 0.17, title, ha='center', va='center', fontsize=8,
                fontweight='bold', color=COLORS['black'])
        ax.plot([x0 + 0.1, x0 + bw - 0.1], [y0 + bh - 0.33] * 2, color=COLORS['blue'], lw=0.6, alpha=0.6)
        rows = [('目标 $f$：', f, COLORS['black']), ('约束 $g$：', g, COLORS['green']),
                ('$\\lambda$：', lam, COLORS['red'])]
        for r, (key, val, col) in enumerate(rows):
            yy = y0 + bh - 0.52 - 0.22 * r
            ax.text(x0 + 0.08, yy, key, ha='left', va='center', fontsize=fs, color=col)
            ax.text(x0 + 0.56, yy, val, ha='left', va='center', fontsize=fs, color=COLORS['black'])
        ax.text(x0 + 0.08, y0 + 0.12, f'一阶条件：{cond}', ha='left', va='center', fontsize=6.6,
                color=COLORS['gray'])

    # 列标题
    ax.text(x_left + bw / 2, y_top + 0.24, '物理与控制', ha='center', va='center', fontsize=9,
            fontweight='bold', color=COLORS['gray'])
    ax.text(x_right + bw / 2, y_top + 0.24, '学习与博弈', ha='center', va='center', fontsize=9,
            fontweight='bold', color=COLORS['gray'])

    # 中心框
    ax.add_patch(FancyBboxPatch((cx - cb_w / 2, cb_y0), cb_w, cb_h,
                                boxstyle='round,pad=0.03,rounding_size=0.1',
                                facecolor='#FFF4D6', edgecolor=COLORS['gold'], lw=1.6))
    ax.text(cx, cb_y0 + cb_h - 0.36, '$\\mathcal{L}=f+\\lambda^{\\top}g$', ha='center', va='center',
            fontsize=13, color=COLORS['black'])
    ax.text(cx, cb_y0 + cb_h - 0.7, '目标 + 价格 × 约束', ha='center', va='center', fontsize=8)
    ax.text(cx, cb_y0 + 0.5, '$\\partial\\mathcal{L}/\\partial x=0$', ha='center', va='center', fontsize=7.6)
    ax.text(cx, cb_y0 + 0.3, '$\\lambda\\geq 0,\\ \\lambda^{\\top}g=0$', ha='center', va='center', fontsize=7.6)
    ax.text(cx, cb_y0 + 0.11, '（KKT）', ha='center', va='center', fontsize=6.6, color=COLORS['gray'])
    # 三问（中心框上方）
    ax.text(cx, cb_y0 + cb_h + 1.0, '每章都问三件事', ha='center', va='center', fontsize=8.2,
            fontweight='bold', color=COLORS['black'])
    for j, (q, col) in enumerate([('目标 $f$ 是什么？', COLORS['black']),
                                  ('约束 $g$ 是什么？', COLORS['green']),
                                  ('$\\lambda$ 代表什么？', COLORS['red'])]):
        ax.text(cx, cb_y0 + cb_h + 0.76 - 0.21 * j, q, ha='center', va='center', fontsize=7.6, color=col)
    # 底注（中心框下方）
    ax.text(cx, cb_y0 - 0.32, '$\\lambda$ 的身份随章节变化，', ha='center', va='center', fontsize=7.4, color=COLORS['gray'])
    ax.text(cx, cb_y0 - 0.52, '结构从未改变', ha='center', va='center', fontsize=7.4, color=COLORS['gray'])

    # 侧框 + 树状连线：每个框 → 竖直母线 → 中心框
    line_kw = dict(color=COLORS['gray'], lw=0.9, solid_capstyle='round')
    for col_items, x0, side in ((left, x_left, -1), (right, x_right, +1)):
        x_edge = x0 + bw if side < 0 else x0                  # 框的内侧边
        x_bus = cx + side * (cb_w / 2 + 0.24)                 # 母线位置
        ys = []
        for i, item in enumerate(col_items):
            y0 = y_boxes[i]
            draw_box(x0, y0, *item)
            yc = y0 + bh / 2
            ax.plot([x_edge, x_bus], [yc, yc], **line_kw)
            ys.append(yc)
        ax.plot([x_bus, x_bus], [min(ys), max(ys)], **line_kw)
        y_mid = cb_y0 + cb_h / 2
        ax.annotate('', xy=(cx + side * cb_w / 2, y_mid), xytext=(x_bus, y_mid),
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['gray'], lw=1.2,
                                    mutation_scale=10, shrinkA=0, shrinkB=1))
    save_figure(fig, f'{OUT}/unified_view')


# =============================================================================
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = {'cbf': fig_cbf_controller, 'multi': fig_multi_obstacle,
            'physics': fig_differentiable_physics, 'latent': fig_latent_dynamics,
            'unified': fig_unified_view}
    wanted = sys.argv[1:] or list(jobs)
    for name in wanted:
        print(f'==== {name} ====')
        jobs[name]()
