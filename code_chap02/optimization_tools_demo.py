"""
第2章扩展：Python 优化工具包演示（figs_chap02 的 4 张性能/结果图）
=====================================================================

运行（仓库根目录）::

    python3 code_chap02/optimization_tools_demo.py

生成：
    figs_chap02/lp_performance.{pdf,png}      线性规划求解时间 vs 变量数（HiGHS）
    figs_chap02/qp_performance.{pdf,png}      投资组合 QP：求解时间与最小风险 vs 资产数（SLSQP）
    figs_chap02/portfolio_weights.{pdf,png}   10 种资产的最优权重
    figs_chap02/nonconvex_comparison.{pdf,png} Rosenbrock：4 种方法的时间与函数评估次数 vs 维度

所有数据都来自脚本里真实的求解（固定随机种子，可复现）；计时取多次重复的中位数。
绘图统一使用 code/textbook_style.py 的教科书风格（中文标签、面板编号、无大标题）。
"""

import os
import sys
import time
import warnings

# 计时基准用单线程 BLAS：多线程 BLAS 在这些小/中等规模的稠密矩阵上反而因线程争用
# 慢一到两个数量级（SLSQP n=500 从 ~10 s 变成 ~5 min），且结果不稳定。必须在 import numpy 之前设置。
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
from scipy.optimize import minimize, linprog  # noqa: E402

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings('ignore')
setup_style()

N_REPEAT = 5          # 计时重复次数（取中位数）
SEED = 42


def _median_time(fn, n_repeat=N_REPEAT):
    """重复调用 fn()，返回 (最后一次的结果, 用时中位数)。"""
    ts, out = [], None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def _fit_power(sizes, times, start=0):
    """在 log-log 坐标下做直线拟合，返回指数 p（t ∝ n^p）。"""
    s, t = np.log(np.asarray(sizes[start:], float)), np.log(np.asarray(times[start:], float))
    return float(np.polyfit(s, t, 1)[0])


# =============================================================================
# 示例1：线性规划 - 资源分配问题
#   max c'x  s.t.  A x <= b, x >= 0   （c, A, b 均为正：利润、资源消耗、资源总量）
#   linprog 做最小化，所以传 -c。若传 +c，最优解恒为 x=0，求解器 0 次迭代就退出，
#   计时没有意义——这是旧版本脚本的问题。
# =============================================================================
def make_lp(n_vars, n_constraints, seed=SEED):
    rng = np.random.RandomState(seed)
    c = rng.rand(n_vars)                       # 单位利润
    A_ub = rng.rand(n_constraints, n_vars)     # 单位资源消耗
    b_ub = rng.rand(n_constraints) * n_vars    # 资源总量
    return c, A_ub, b_ub


def solve_linear_program_scipy(n_vars, n_constraints):
    """使用 scipy.optimize.linprog (HiGHS) 求解线性规划，返回 (最优值, 迭代次数, 用时)。"""
    c, A_ub, b_ub = make_lp(n_vars, n_constraints)
    res, elapsed = _median_time(lambda: linprog(-c, A_ub=A_ub, b_ub=b_ub, method='highs'))
    assert res.success, res.message
    return -res.fun, res.nit, elapsed


# =============================================================================
# 示例2：二次规划 - 投资组合优化
#   min ½ w'Σw  s.t.  1'w = 1, μ'w >= r_target, 0 <= w <= 1
# =============================================================================
def make_portfolio(n_assets, seed=SEED):
    rng = np.random.RandomState(seed)
    A = rng.rand(n_assets, n_assets)
    Sigma = A @ A.T + np.eye(n_assets) * 0.1   # 协方差矩阵（正定）
    mu = rng.rand(n_assets) * 0.1 + 0.05       # 预期收益
    return Sigma, mu


def portfolio_optimization_scipy(n_assets, target_return=0.08):
    """SLSQP 求解投资组合优化，返回 (最小风险, 用时, 权重, 均匀分配的风险, 迭代次数)。"""
    Sigma, mu = make_portfolio(n_assets)

    def objective(w):
        return 0.5 * w @ Sigma @ w

    def grad(w):
        return Sigma @ w

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones_like(w)},
        {'type': 'ineq', 'fun': lambda w: w @ mu - target_return, 'jac': lambda w: mu},
    ]
    bounds = [(0, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    res, elapsed = _median_time(lambda: minimize(objective, w0, method='SLSQP', jac=grad,
                                                 constraints=constraints, bounds=bounds),
                                n_repeat=N_REPEAT if n_assets <= 200 else 2)
    assert res.success, res.message
    return res.fun, elapsed, res.x, objective(w0), res.nit


# =============================================================================
# 示例3：非凸优化 - Rosenbrock函数
# =============================================================================
def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


METHODS = ['Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell']
DIMS = [2, 5, 10, 20]


def rosenbrock_optimization(methods=METHODS, dims=DIMS):
    """从 x0 = 0 出发，用 4 种方法最小化 Rosenbrock 函数。"""
    results = {}
    for dim in dims:
        results[dim] = {}
        x0 = np.zeros(dim)
        for method in methods:
            res, elapsed = _median_time(
                lambda: minimize(rosenbrock, x0, method=method, options={'maxiter': 10000}))
            converged = bool(res.success) and res.fun < 1e-6
            results[dim][method] = {'value': res.fun, 'time': elapsed,
                                    'success': converged, 'nfev': res.nfev}
            print(f'  dim={dim:2d} {method:12s}: f={res.fun:.2e} nfev={res.nfev:5d} '
                  f't={elapsed * 1e3:7.2f} ms  {"收敛" if converged else "未收敛"}')
    return results


# =============================================================================
# 性能测试
# =============================================================================
def benchmark_linear_programming():
    sizes = [10, 50, 100, 200, 500, 1000]
    times, nits = [], []
    for n in sizes:
        val, nit, t = solve_linear_program_scipy(n, n // 2)
        times.append(t); nits.append(nit)
        print(f'  LP: n={n:5d}, m={n // 2:4d}, 最优值={val:.3f}, 迭代={nit:4d}, time={t * 1e3:.2f} ms')
    return sizes, times, nits


def benchmark_quadratic_programming():
    sizes = [10, 20, 50, 100, 200, 500]
    times, risks, risks_eq = [], [], []
    for n in sizes:
        risk, t, _, risk_eq, nit = portfolio_optimization_scipy(n)
        times.append(t); risks.append(risk); risks_eq.append(risk_eq)
        print(f'  QP: n={n:4d}, 迭代={nit:3d}, time={t:.4f}s, 最小风险={risk:.4f}, 均匀分配风险={risk_eq:.4f}')
    return sizes, times, risks, risks_eq


# =============================================================================
# 绘图（教科书风格）
# =============================================================================
def plot_lp_performance(sizes, times, nits, out='figs_chap02/lp_performance'):
    """线性规划：求解时间 vs 变量数（log-log），并标出经验幂律指数。"""
    fig, ax = plt.subplots(figsize=fig_size(1, width=4.4, aspect=0.68))
    ax.loglog(sizes, times, 'o-', color=COLORS['blue'], label='scipy.optimize.linprog（HiGHS）')
    p = _fit_power(sizes, times, start=2)
    ref = times[-1] * (np.array(sizes[2:]) / sizes[-1]) ** p
    ax.loglog(sizes[2:], ref, ls='--', color=COLORS['gray'], lw=1.2,
              label=f'拟合 $t\\propto n^{{{p:.1f}}}$（$n\\geq100$）')
    for n, t, k in zip(sizes, times, nits):
        ax.annotate(f'{k} 次迭代', (n, t), textcoords='offset points', xytext=(7, -9),
                    ha='left', va='top', fontsize=7.5, color=COLORS['gray'])
    ax.annotate(f'{times[-1] * 1e3:.0f} ms', (sizes[-1], times[-1]), textcoords='offset points',
                xytext=(-8, 4), ha='right', va='bottom', fontsize=8.5, color=COLORS['blue'])
    ax.set_ylim(min(times) * 0.4, max(times) * 3)
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.6)
    ax.set_xlabel('变量数 $n$（约束数 $n/2$）')
    ax.set_ylabel('求解时间（秒）')
    ax.legend(loc='upper left')
    save_figure(fig, out)


def plot_qp_performance(sizes, times, risks, risks_eq, out='figs_chap02/qp_performance'):
    """二次规划：(a) 求解时间 vs 资产数；(b) 最小风险 vs 资产数（与均匀分配对照）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(2, aspect=0.75))
    ax1.loglog(sizes, times, 'o-', color=COLORS['blue'], label='SLSQP 求解时间')
    p = _fit_power(sizes, times, start=2)
    ref = times[-1] * (np.array(sizes[2:]) / sizes[-1]) ** p
    ax1.loglog(sizes[2:], ref, ls='--', color=COLORS['gray'], lw=1.2,
               label=f'拟合 $t\\propto n^{{{p:.1f}}}$（$n\\geq50$）')
    ax1.annotate(f'{times[-1]:.1f} s', (sizes[-1], times[-1]), textcoords='offset points',
                 xytext=(-8, 4), ha='right', va='bottom', fontsize=8.5, color=COLORS['blue'])
    ax1.set_ylim(min(times) * 0.4, max(times) * 4)
    ax1.set_xlabel('资产数 $n$')
    ax1.set_ylabel('求解时间（秒）')
    ax1.legend(loc='upper left')
    panel_label(ax1, '(a)')

    ax2.plot(sizes, risks_eq, 's--', color=COLORS['gray'], label='均匀分配 $w_i=1/n$')
    ax2.plot(sizes, risks, 'o-', color=COLORS['blue'], label='最优组合 $\\min\\ \\frac{1}{2} w^\\top\\Sigma w$')
    ax2.set_xlabel('资产数 $n$')
    ax2.set_ylabel('组合风险 $\\frac{1}{2} w^\\top\\Sigma w$')
    ax2.legend(loc='upper left')
    panel_label(ax2, '(b)')
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, out)


def plot_portfolio_weights(out='figs_chap02/portfolio_weights'):
    """10 种资产的最优权重柱状图，与均匀分配 1/n 对照。"""
    n = 10
    risk, _, w, risk_eq, _ = portfolio_optimization_scipy(n)
    Sigma, mu = make_portfolio(n)
    print(f'  最优权重: {np.round(w, 3)}')
    print(f'  组合风险: {risk:.4f}（均匀分配 {risk_eq:.4f}）, 组合收益 {w @ mu:.4f}（目标 0.08）')

    fig, ax = plt.subplots(figsize=fig_size(1, width=4.4, aspect=0.68))
    idx = np.arange(1, n + 1)
    ax.bar(idx, w, width=0.65, color=COLORS['blue'], label='最优权重 $w_i^*$')
    ax.axhline(1 / n, color=COLORS['gray'], ls='--', lw=1.2, label=f'均匀分配 $1/n={1 / n:.1f}$')
    for i, wi in zip(idx, w):
        if wi > 0.005:
            ax.text(i, wi + 0.008, f'{wi:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'资产{i}' for i in idx], fontsize=8)
    ax.set_xlabel('资产')
    ax.set_ylabel('权重 $w_i$')
    ax.set_ylim(0, max(w) * 1.25)
    ax.grid(axis='x', visible=False)
    ax.text(0.98, 0.72,
            f'风险：最优 {risk:.2f}，均匀分配 {risk_eq:.2f}\n收益 $\\mu^\\top w^*={w @ mu:.3f}\\geq 0.08$（不起作用）',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    save_figure(fig, out)


METHOD_LABELS = {
    'Nelder-Mead': 'Nelder–Mead（单纯形，无梯度）',
    'BFGS': 'BFGS（拟牛顿，差分梯度）',
    'L-BFGS-B': 'L-BFGS-B（有限内存拟牛顿）',
    'Powell': 'Powell（方向集，无梯度）',
}
METHOD_STYLE = {
    'Nelder-Mead': (COLORS['orange'], 'o'),
    'BFGS': (COLORS['blue'], 's'),
    'L-BFGS-B': (COLORS['green'], '^'),
    'Powell': (COLORS['purple'], 'D'),
}


def plot_nonconvex_comparison(results, out='figs_chap02/nonconvex_comparison'):
    """Rosenbrock：(a) 求解时间 vs 维度；(b) 函数评估次数 vs 维度。空心标记 = 未收敛。"""
    dims = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(2, aspect=0.82))
    handles = []
    for method in METHODS:
        color, marker = METHOD_STYLE[method]
        t = np.array([results[d][method]['time'] for d in dims])
        nf = np.array([results[d][method]['nfev'] for d in dims])
        ok = np.array([results[d][method]['success'] for d in dims])
        for ax, y in ((ax1, t), (ax2, nf)):
            h, = ax.plot(dims, y, '-', color=color, marker=marker, mfc=color, label=METHOD_LABELS[method])
            if (~ok).any():
                ax.plot(np.array(dims)[~ok], y[~ok], ls='none', marker=marker, mfc='white',
                        mec=color, mew=1.4, ms=6.5, zorder=5)
        handles.append(h)
    for ax in (ax1, ax2):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.minorticks_off()
        ax.set_xlabel('维度 $n$')
    ax1.set_ylabel('求解时间（秒）')
    ax2.set_ylabel('函数评估次数')
    ax2.text(0.03, 0.97, '空心标记：未收敛到 $f<10^{-6}$', transform=ax2.transAxes,
             ha='left', va='top', fontsize=7.5, color=COLORS['gray'])
    panel_label(ax1, '(a)')
    panel_label(ax2, '(b)')
    # 图例放在两个面板下方，不遮挡曲线
    fig.legend(handles, [METHOD_LABELS[m] for m in METHODS], loc='lower center', ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=2.0)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.13, 1, 1))
    save_figure(fig, out)


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("[1] 线性规划性能测试")
    lp_sizes, lp_times, lp_nits = benchmark_linear_programming()
    plot_lp_performance(lp_sizes, lp_times, lp_nits)

    print("\n[2] 二次规划性能测试")
    qp_sizes, qp_times, qp_risks, qp_risks_eq = benchmark_quadratic_programming()
    plot_qp_performance(qp_sizes, qp_times, qp_risks, qp_risks_eq)

    print("\n[3] 投资组合权重")
    plot_portfolio_weights()

    print("\n[4] 非凸优化方法比较")
    rosenbrock_results = rosenbrock_optimization()
    plot_nonconvex_comparison(rosenbrock_results)

    print("\n" + "=" * 60)
    print("全部完成。")
