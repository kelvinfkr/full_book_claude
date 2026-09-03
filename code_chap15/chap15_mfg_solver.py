"""
第15章：一维平均场博弈（MFG）的真实数值求解 —— figs_chap15/mfg_solution.pdf
=====================================================================

正文（chap15.tex "MFG 的数学方程"）里的方程组，取 L(x,v,m) = ½v² + F(x,m)：

    HJB（倒向）  -∂_t u - ν ∂_xx u + ½|∂_x u|² = F(x, m),   u(x,T) = g(x)
    FP （正向）   ∂_t m - ν ∂_xx m - ∂_x(m ∂_x u) = 0,        m(x,0) = m₀(x)
    最优速度      v* = -∂_p H = -∂_x u

耦合项（"群体想去某处，但讨厌拥挤"）：

    F(x, m) = (α/2)(x - x_target)² + κ m(x,t)

第一项是位置偏好（离目标越远越贵），第二项是拥挤代价（当地人越多越贵）。
κ m 正是"别人对我的价格"——本章 λ 的平均场版本。

数值方法
--------
* 均匀网格 x ∈ [-L, L]，Nx 个点；时间 [0, T]，Nt 步。
* HJB：倒向时间推进，扩散项隐式（三对角求解），Hamilton 项用
  Osher–Sethian 单调迎风格式显式处理（需满足 CFL：max|∂_x u|·dt/h ≤ 1，脚本会打印）。
* FP：正向时间推进，迎风通量 + 隐式扩散，守恒形式（离散总质量精确守恒，
  隐式迎风矩阵是 M 矩阵，密度保持非负）。速度取自 HJB 的离散解 u^{n+1}，
  与 HJB 格式互为伴随。
* 外层不动点迭代：m → (HJB) → u → (FP) → m̃，阻尼更新 m ← (1-θ)m + θm̃，
  记录残差 ‖m̃ - m‖_∞ 与 ‖u_k - u_{k-1}‖_∞ 直到 < tol。

运行（仓库根目录）::

    python3 code_chap15/chap15_mfg_solver.py

输出 figs_chap15/mfg_solution.{pdf,png}，并在终端打印每次迭代的残差。
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import solve_banded

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# 问题参数
# ---------------------------------------------------------------------------
class MFGProblem:
    def __init__(self, L=2.5, Nx=251, T=1.5, Nt=1500, nu=0.05,
                 alpha=3.0, kappa=2.0, x_target=1.0, x0=-1.0, sigma0=0.3):
        self.L, self.Nx, self.T, self.Nt = L, Nx, T, Nt
        self.nu, self.alpha, self.kappa = nu, alpha, kappa
        self.x_target, self.x0, self.sigma0 = x_target, x0, sigma0
        self.x = np.linspace(-L, L, Nx)
        self.h = self.x[1] - self.x[0]
        self.t = np.linspace(0, T, Nt + 1)
        self.dt = T / Nt
        # 初始密度：高斯，归一化到 Σ m h = 1
        m0 = np.exp(-(self.x - x0) ** 2 / (2 * sigma0 ** 2))
        self.m0 = m0 / (m0.sum() * self.h)
        # 终端代价：与位置偏好同形
        self.g = 0.5 * alpha * (self.x - x_target) ** 2

    def F(self, m):
        """运行代价中的耦合项 F(x, m) = (α/2)(x - x_t)² + κ m。"""
        return 0.5 * self.alpha * (self.x - self.x_target) ** 2 + self.kappa * m


# ---------------------------------------------------------------------------
# 离散算子
# ---------------------------------------------------------------------------
def _neumann_laplacian_banded(Nx, h, coef):
    """返回 (I - coef·D²) 的三对角 banded 形式（Neumann 边界，ghost 点法）。"""
    r = coef / h ** 2
    ab = np.zeros((3, Nx))
    ab[0, 1:] = -r            # 上对角
    ab[1, :] = 1 + 2 * r      # 主对角
    ab[2, :-1] = -r           # 下对角
    ab[1, 0] = 1 + r          # 边界：ghost u_{-1}=u_0
    ab[1, -1] = 1 + r
    return ab


def hjb_backward(P, m):
    """给定密度 m[n, i]，倒向求解 HJB，返回 u[n, i] 与 CFL 数。"""
    Nx, Nt, h, dt, nu = P.Nx, P.Nt, P.h, P.dt, P.nu
    u = np.empty((Nt + 1, Nx))
    u[-1] = P.g
    ab = _neumann_laplacian_banded(Nx, h, nu * dt)
    cfl = 0.0
    for n in range(Nt - 1, -1, -1):
        un1 = u[n + 1]
        # 单侧差分（边界处 Neumann：D⁻u_0 = D⁺u_{N-1} = 0）
        Dm = np.zeros(Nx); Dm[1:] = (un1[1:] - un1[:-1]) / h
        Dp = np.zeros(Nx); Dp[:-1] = (un1[1:] - un1[:-1]) / h
        # Osher–Sethian 迎风 Hamilton 量：H(p)=½p² → ½[max(D⁻u,0)² + min(D⁺u,0)²]
        Hnum = 0.5 * (np.maximum(Dm, 0) ** 2 + np.minimum(Dp, 0) ** 2)
        cfl = max(cfl, np.abs(Dm).max() * dt / h)
        rhs = un1 - dt * Hnum + dt * P.F(m[n])
        u[n] = solve_banded((1, 1), ab, rhs)
    return u, cfl


def fp_forward(P, u):
    """给定 u[n, i]，正向求解 FP（迎风通量 + 隐式扩散，守恒形式），返回 m[n, i]。"""
    Nx, Nt, h, dt, nu = P.Nx, P.Nt, P.h, P.dt, P.nu
    m = np.empty((Nt + 1, Nx))
    m[0] = P.m0
    r = nu * dt / h ** 2
    for n in range(Nt):
        # 半网格点速度 v_{i+1/2} = -(u_{i+1}-u_i)/h，取自 u^{n+1}（与 HJB 格式伴随）
        v = -(u[n + 1, 1:] - u[n + 1, :-1]) / h          # 长度 Nx-1
        vp, vm = np.maximum(v, 0), np.minimum(v, 0)
        ab = np.zeros((3, Nx))
        diag = np.full(Nx, 1 + 2 * r)
        diag[0] = 1 + r; diag[-1] = 1 + r
        # 通量 F_{i+1/2} = m_i v⁺ + m_{i+1} v⁻；行 i：+(dt/h)(F_{i+1/2} - F_{i-1/2})
        diag[:-1] += dt / h * vp          # 来自 F_{i+1/2} 中的 m_i
        diag[1:] -= dt / h * vm           # 来自 -F_{i-1/2} 中的 m_i
        upper = -r * np.ones(Nx - 1) + dt / h * vm     # m_{i+1} 的系数
        lower = -r * np.ones(Nx - 1) - dt / h * vp     # m_{i-1} 的系数
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        m[n + 1] = solve_banded((1, 1), ab, m[n])
    return m


def solve_mfg(P, theta=0.3, tol=1e-8, max_iter=300, verbose=True, adaptive=True):
    """外层阻尼不动点迭代 m ← (1-θ)m + θ·FP(HJB(m))。

    阻尼是必要的：拥挤项 κm 让"密度→最优响应→新密度"这个映射带有负反馈
    （人多的地方大家都想离开），不加阻尼会周期性振荡。adaptive=True 时若残差
    不降反升就把 θ 缩小 0.7 倍（保底 0.05）。返回 (u, m, 残差历史 m, 残差历史 u, θ 历史)。
    """
    m = np.tile(P.m0, (P.Nt + 1, 1))         # 初猜：密度不动
    u_prev = None
    hist_m, hist_u, hist_theta = [], [], []
    t0 = time.time()
    for k in range(1, max_iter + 1):
        u, cfl = hjb_backward(P, m)
        m_new = fp_forward(P, u)
        res_m = np.abs(m_new - m).max()
        res_u = np.abs(u - u_prev).max() if u_prev is not None else np.nan
        if adaptive and hist_m and res_m > hist_m[-1]:
            theta = max(0.7 * theta, 0.05)
        hist_m.append(res_m); hist_u.append(res_u); hist_theta.append(theta)
        mass_err = np.abs(m_new.sum(axis=1) * P.h - 1).max()
        if verbose:
            print(f'  迭代 {k:3d}: ‖m̃-m‖∞ = {res_m:.3e}   ‖u_k-u_(k-1)‖∞ = {res_u:.3e}'
                  f'   θ = {theta:.3f}   质量误差 = {mass_err:.1e}   CFL = {cfl:.2f}'
                  f'   min m = {m_new.min():.1e}')
        m = (1 - theta) * m + theta * m_new
        u_prev = u
        if res_m < tol:
            break
    if verbose:
        print(f'  {"收敛" if res_m < tol else "未收敛"}：{k} 次迭代，用时 {time.time() - t0:.1f} s')
    # 收敛后再做一次 HJB 使 (u, m) 严格自洽
    u, _ = hjb_backward(P, m)
    return u, m, np.array(hist_m), np.array(hist_u), np.array(hist_theta)


def drift_trajectories(P, u, starts):
    """沿最优速度场 ẋ = -∂_x u(x,t) 积分（无噪声的"平均个体"轨迹）。"""
    ux = np.gradient(u, P.h, axis=1)
    paths = []
    for xs in starts:
        xp = [xs]
        xc = xs
        for n in range(P.Nt):
            vel = np.interp(xc, P.x, -ux[n])
            xc = np.clip(xc + P.dt * vel, -P.L, P.L)
            xp.append(xc)
        paths.append(np.array(xp))
    return paths


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_solution(P, u, m, hist_m, hist_u, m_nocrowd, out='figs_chap15/mfg_solution'):
    setup_style()
    x, t = P.x, P.t
    X, Tm = np.meshgrid(x, t)
    fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2, aspect=0.8))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    xlim = (-2.0, 2.0)

    # (a) 密度 m(x,t) + 平均个体轨迹
    cf = ax_a.contourf(X, Tm, m, levels=24, cmap='Blues')
    cb = fig.colorbar(cf, ax=ax_a, pad=0.02)
    cb.set_label('密度 $m(x,t)$')
    cb.ax.tick_params(labelsize=7.5)
    paths = drift_trajectories(P, u, [-1.5, -1.2, -0.9, -0.6, -0.3])
    for i, p in enumerate(paths):
        ax_a.plot(p, t, color=COLORS['red'], lw=1.0, ls='--',
                  label='个体最优轨迹 $\\dot{x}=-\\partial_x u$' if i == 0 else None)
    ax_a.axvline(P.x_target, color=COLORS['black'], lw=0.8, ls=':', label='目标点 $x_{\\rm target}$')
    ax_a.set_xlim(xlim); ax_a.set_ylim(0, P.T)
    ax_a.set_xlabel('位置 $x$'); ax_a.set_ylabel('时间 $t$')
    ax_a.grid(False)
    ax_a.legend(loc='upper left', fontsize=7.5, handlelength=1.6)
    panel_label(ax_a, '(a)')

    # (b) 价值函数 u(x,t) + 最优速度场箭头
    cf = ax_b.contourf(X, Tm, u, levels=24, cmap='Oranges')
    cb = fig.colorbar(cf, ax=ax_b, pad=0.02)
    cb.set_label('价值函数 $u(x,t)$')
    cb.ax.tick_params(labelsize=7.5)
    ux = np.gradient(u, P.h, axis=1)
    vel = -ux
    sx = slice(None, None, 20); st = slice(80, P.Nt - 180, 160)   # 顶部留空给注记
    ax_b.quiver(X[st, sx], Tm[st, sx], vel[st, sx], np.zeros_like(vel[st, sx]),
                color=COLORS['black'], scale=45, width=0.006, headwidth=4, headlength=5,
                pivot='tail')
    ax_b.axvline(P.x_target, color=COLORS['black'], lw=0.8, ls=':')
    ax_b.text(0.03, 0.96, '箭头：最优速度 $v^*=-\\partial_x u$', transform=ax_b.transAxes,
              fontsize=7.5, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax_b.set_xlim(xlim); ax_b.set_ylim(0, P.T)
    ax_b.set_xlabel('位置 $x$'); ax_b.set_ylabel('时间 $t$')
    ax_b.grid(False)
    panel_label(ax_b, '(b)')

    # (c) 若干时刻的密度剖面 + 无拥挤惩罚对照
    times = [0.0, 0.5, 1.0, P.T]
    shades = plt.cm.Blues(np.linspace(0.4, 1.0, len(times)))
    for ti, c in zip(times, shades):
        n = int(round(ti / P.dt))
        if ti == 0.0:
            lab = '$t=0$（初始）'
        elif ti == P.T:
            lab = f'$t=T$，$\\kappa={P.kappa:g}$'
        else:
            lab = f'$t={ti:g}$'
        ax_c.plot(x, m[n], color=c, label=lab)
    nT = P.Nt
    ax_c.plot(x, m_nocrowd[nT], color=COLORS['gray'], ls='--', lw=1.3,
              label='$t=T$，$\\kappa=0$（无拥挤惩罚）')
    ax_c.axvline(P.x_target, color=COLORS['black'], lw=0.8, ls=':')
    ax_c.set_xlim(xlim)
    ax_c.set_ylim(0, 1.3 * max(m_nocrowd[nT].max(), m.max()))
    ax_c.set_xlabel('位置 $x$'); ax_c.set_ylabel('密度 $m(x,t)$')
    ax_c.legend(fontsize=7.5, loc='upper left', handlelength=1.6)
    panel_label(ax_c, '(c)')

    # (d) 不动点迭代残差
    it = np.arange(1, len(hist_m) + 1)
    ax_d.semilogy(it, hist_m, 'o-', color=COLORS['blue'], ms=3.5, markevery=5,
                  label='$\\|\\tilde m_k - m_k\\|_\\infty$（密度）')
    ax_d.semilogy(it[1:], hist_u[1:], 's-', color=COLORS['orange'], ms=3.5, markevery=5,
                  label='$\\|u_k - u_{k-1}\\|_\\infty$（价值函数）')
    ax_d.axhline(1e-8, color=COLORS['gray'], ls=':', lw=1.0)
    ax_d.text(1, 1.6e-8, '收敛阈值 $10^{-8}$', ha='left', va='bottom',
              fontsize=7.5, color=COLORS['gray'])
    ax_d.set_xlabel('不动点迭代次数 $k$')
    ax_d.set_ylabel('残差')
    ax_d.legend(fontsize=7.5, loc='upper right', handlelength=1.6)
    panel_label(ax_d, '(d)')

    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    save_figure(fig, out)


def main():
    P = MFGProblem()
    print(f'网格：Nx={P.Nx}, h={P.h:.3f}; Nt={P.Nt}, dt={P.dt:.4f}; ν={P.nu}, α={P.alpha}, '
          f'κ={P.kappa}, x_target={P.x_target}, x0={P.x0}, σ0={P.sigma0}')
    print('求解带拥挤惩罚的 MFG（κ = %.2f）：' % P.kappa)
    u, m, hist_m, hist_u, hist_theta = solve_mfg(P, theta=0.3, tol=1e-8)
    print('求解无拥挤惩罚的对照（κ = 0，HJB 与 m 解耦，一次迭代即收敛）：')
    P0 = MFGProblem(kappa=0.0)
    _, m0, _, _, _ = solve_mfg(P0, theta=1.0, tol=1e-8, adaptive=False)
    # 打印几个可核对的数字
    nT = P.Nt
    mean_T = (P.x * m[nT]).sum() * P.h
    std_T = np.sqrt(((P.x - mean_T) ** 2 * m[nT]).sum() * P.h)
    mean0 = (P0.x * m0[nT]).sum() * P0.h
    std0 = np.sqrt(((P0.x - mean0) ** 2 * m0[nT]).sum() * P0.h)
    print(f'κ={P.kappa}: t=T 时密度均值 {mean_T:.3f}，标准差 {std_T:.3f}，峰值 {m[nT].max():.3f}')
    print(f'κ=0  : t=T 时密度均值 {mean0:.3f}，标准差 {std0:.3f}，峰值 {m0[nT].max():.3f}')
    os.makedirs('figs_chap15', exist_ok=True)
    plot_solution(P, u, m, hist_m, hist_u, m0)


if __name__ == '__main__':
    main()
