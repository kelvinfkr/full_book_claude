#!/usr/bin/env python3
"""
第13章：有限元与性能实验演示
============================

从仓库根目录运行::

    python3 code/chap13_performance.py          # 跑全部实验（GIL / 缓存 / NumPy / 有限元 / 预条件子）
    python3 code/chap13_performance.py fem      # 只重绘有限元收敛图 figs_chap13/chap13_fig2

实验内容：
1. Python GIL 对多线程的影响                  -> figs/chap13_fig1.png（旧图，未重绘）
2. 按行访问 vs 按列访问的速度差异
3. NumPy vs 纯Python的性能对比
4. 一维线性有限元：解、刚度矩阵、收敛阶       -> figs_chap13/chap13_fig2.{pdf,png}
5. 预条件子对共轭梯度法收敛的影响             -> figs/chap13_fig3.png（旧图，未重绘）
"""
import sys
import time
import threading

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

setup_style()


# =============================================================================
# 实验1-3：性能实验（fig1）
# =============================================================================
def run_performance_experiments():
    print("=" * 60)
    print("实验1：Python GIL 对多线程的影响")
    print("=" * 60)

    def count_small():
        n = 0
        for _ in range(5_000_000):
            n += 1
        return n

    t0 = time.time(); count_small(); single_time = time.time() - t0
    print(f"单线程耗时: {single_time:.2f} 秒")
    t0 = time.time()
    t1, t2 = threading.Thread(target=count_small), threading.Thread(target=count_small)
    t1.start(); t2.start(); t1.join(); t2.join()
    double_time = time.time() - t0
    print(f"双线程耗时: {double_time:.2f} 秒（GIL 使多线程无法并行执行计算密集型任务）")

    print("\n实验2：按行访问 vs 按列访问（Cache效应）")
    N = 2000
    A = np.random.rand(N, N)
    t0 = time.time(); s = 0.0
    for i in range(N):
        for j in range(N):
            s += A[i, j]
    row_time = time.time() - t0
    t0 = time.time(); s = 0.0
    for j in range(N):
        for i in range(N):
            s += A[i, j]
    col_time = time.time() - t0
    print(f"按行访问: {row_time:.2f} 秒, 按列访问: {col_time:.2f} 秒, 比值 {col_time / row_time:.2f}x")

    print("\n实验3：NumPy vs 纯Python")
    size = 1_000_000
    x, y = np.random.rand(size), np.random.rand(size)
    t0 = time.time(); z1 = [x[i] + y[i] for i in range(size)]; python_time = time.time() - t0
    t0 = time.time(); z2 = x + y; numpy_time = time.time() - t0
    speedup = python_time / numpy_time
    print(f"纯 Python: {python_time:.3f} 秒, NumPy: {numpy_time:.3f} 秒, 加速 {speedup:.1f} 倍")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1 = axes[0]
    bars1 = ax1.bar(['单线程', '双线程'], [single_time, double_time], color=['steelblue', 'coral'])
    ax1.set_ylabel('时间 (秒)', labelpad=10)
    ax1.set_title('实验1: Python GIL效应\n(多线程不能加速CPU密集型任务)')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.2f}s', ha='center', va='bottom')
    ax2 = axes[1]
    bars2 = ax2.bar(['按行访问(快)', '按列访问(慢)'], [row_time, col_time], color=['green', 'red'])
    ax2.set_ylabel('时间 (秒)', labelpad=10)
    ax2.set_title('实验2: 内存访问模式\n(按行访问利用Cache局部性更快)')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.2f}s', ha='center', va='bottom')
    ax3 = axes[2]
    bars3 = ax3.bar(['纯Python', 'NumPy'], [python_time, numpy_time], color=['orange', 'blue'])
    ax3.set_ylabel('时间 (秒)', labelpad=10)
    ax3.set_title(f'实验3: NumPy vs 纯Python\n(NumPy快{speedup:.0f}倍!)')
    for bar in bars3:
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.3f}s', ha='center', va='bottom')
    ax3.set_yscale('log')
    plt.tight_layout()
    fig.savefig('figs/chap13_fig1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("图像已保存到 figs/chap13_fig1.png")


# =============================================================================
# 实验4：一维线性有限元（fig2）
# =============================================================================
def solve_1d_fem(n_elements, f_source=1.0):
    """线性单元求解 -u'' = f（常数）, u(0)=u(1)=0。返回节点坐标、节点值、内部刚度矩阵。"""
    n_nodes = n_elements + 1
    h = 1.0 / n_elements
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    ke = (1 / h) * np.array([[1, -1], [-1, 1]])       # 单元刚度矩阵
    fe = f_source * h / 2 * np.array([1, 1])          # 单元载荷向量
    for e in range(n_elements):
        K[e:e + 2, e:e + 2] += ke
        F[e:e + 2] += fe
    K_in, F_in = K[1:-1, 1:-1], F[1:-1]                # 施加 Dirichlet 边界条件
    u = np.zeros(n_nodes)
    u[1:-1] = np.linalg.solve(K_in, F_in)
    return np.linspace(0, 1, n_nodes), u, K_in


def exact_solution(x):
    return x * (1 - x) / 2


def exact_derivative(x):
    return 0.5 - x


def fem_errors(x_nodes, u_nodes):
    """在每个单元上用 5 点 Gauss 求积计算 L2 误差、H1 半范数误差；最大误差在单元内细采样得到。"""
    gp, gw = np.polynomial.legendre.leggauss(5)
    l2, h1, linf = 0.0, 0.0, 0.0
    for e in range(len(x_nodes) - 1):
        xa, xb = x_nodes[e], x_nodes[e + 1]
        ua, ub = u_nodes[e], u_nodes[e + 1]
        h = xb - xa
        xq = xa + (gp + 1) / 2 * h                       # 映射到单元
        uh = ua + (ub - ua) * (xq - xa) / h              # 线性插值
        duh = (ub - ua) / h
        l2 += np.sum(gw * (uh - exact_solution(xq)) ** 2) * h / 2
        h1 += np.sum(gw * (duh - exact_derivative(xq)) ** 2) * h / 2
        xs = np.linspace(xa, xb, 41)
        linf = max(linf, np.max(np.abs(ua + (ub - ua) * (xs - xa) / h - exact_solution(xs))))
    nodal = np.max(np.abs(u_nodes - exact_solution(x_nodes)))
    return np.sqrt(l2), np.sqrt(h1), linf, nodal


def run_fem():
    print("=" * 60)
    print("有限元方法：1D 热传导问题 -u'' = 1, u(0)=u(1)=0")
    print("=" * 60)
    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) 解的对比
    x_fine = np.linspace(0, 1, 400)
    ax_a.plot(x_fine, exact_solution(x_fine), color=COLORS['black'], lw=2.0, label='精确解 $u=x(1-x)/2$')
    for n_elem, col in ((4, COLORS['blue']), (8, COLORS['orange'])):
        x_fem, u_fem, _ = solve_1d_fem(n_elem)
        ax_a.plot(x_fem, u_fem, 'o-', color=col, ms=3.5, lw=1.2, label=f'有限元 $N={n_elem}$')
    ax_a.set_xlabel('$x$')
    ax_a.set_ylabel('$u(x)$')
    ax_a.set_ylim(0, 0.15)
    ax_a.legend(loc='lower center', fontsize=8)
    panel_label(ax_a, '(a)')

    # (b) 刚度矩阵（N=8，去掉边界节点后 7×7）
    n_show = 8
    _, _, K_in = solve_1d_fem(n_show)
    n = K_in.shape[0]
    ax_b.set_xlim(-0.5, n - 0.5)
    ax_b.set_ylim(n - 0.5, -0.5)
    ax_b.set_aspect('equal')
    for i in range(n):
        for j in range(n):
            v = K_in[i, j]
            if v > 0:
                ax_b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=COLORS['blue'], alpha=0.85))
                ax_b.text(j, i, '$2/h$', ha='center', va='center', fontsize=7, color='white')
            elif v < 0:
                ax_b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=COLORS['red'], alpha=0.75))
                ax_b.text(j, i, '$-1/h$', ha='center', va='center', fontsize=6.5, color='white')
            else:
                ax_b.text(j, i, '0', ha='center', va='center', fontsize=6.5, color=COLORS['gray'])
    ax_b.set_xticks(range(n))
    ax_b.set_yticks(range(n))
    ax_b.set_xticklabels([f'{i + 1}' for i in range(n)], fontsize=7)
    ax_b.set_yticklabels([f'{i + 1}' for i in range(n)], fontsize=7)
    ax_b.set_xlabel(f'列 $j$（$N={n_show}$：非零元 {np.count_nonzero(K_in)}/{n * n}）')
    ax_b.set_ylabel('行 $i$（内部节点）')
    ax_b.grid(False)
    panel_label(ax_b, '(b)')

    # (c) 收敛：L2 / H1 半范数 / 最大误差（单元内） vs h
    n_list = [4, 8, 16, 32, 64, 128, 256]
    h_list = np.array([1.0 / n for n in n_list])
    E = np.array([fem_errors(*solve_1d_fem(n)[:2]) for n in n_list])
    names = ['$L^2$ 误差 $\\|u_h-u\\|_{L^2}$', '$H^1$ 半范数误差 $\\|u_h^{\\prime}-u^{\\prime}\\|_{L^2}$', '最大误差 $\\max_x|u_h(x)-u(x)|$（单元内部）']
    cols = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    mks = ['o', 's', '^']
    for k in range(3):
        slope = np.polyfit(np.log(h_list), np.log(E[:, k]), 1)[0]
        ax_c.loglog(h_list, E[:, k], marker=mks[k], color=cols[k], ms=4, label=f'{names[k]}：斜率 {slope:.2f}')
        print(f"  {names[k]}: 拟合斜率 {slope:.3f}")
    for order, y0, col in ((2, E[0, 0] * 2.5, COLORS['blue']), (1, E[0, 1] * 2.5, COLORS['orange'])):
        ax_c.loglog(h_list, y0 * (h_list / h_list[0]) ** order, ls=':', color=col, lw=0.9)
        ax_c.text(h_list[0] * 1.12, y0, f'$O(h^{order})$', color=col, fontsize=8, va='center', ha='left')
    ax_c.set_xlabel('单元尺寸 $h$')
    ax_c.set_ylabel('误差')
    ax_c.set_xlim(2.5e-3, 0.5)
    ax_c.legend(loc='lower right', fontsize=7.5)
    ax_c.text(0.02, 0.96, f'节点处误差 $\\max_i|u_h(x_i)-u(x_i)|<10^{{-14}}$\n（舍入水平：节点值超收敛，不在此图之列）',
              transform=ax_c.transAxes, ha='left', va='top', fontsize=7)
    panel_label(ax_c, '(c)')

    save_figure(fig, 'figs_chap13/chap13_fig2')
    print(f"  节点误差最大值（所有网格）: {E[:, 3].max():.2e}")
    print("\n刚度矩阵示例（4 个单元，内部 3×3）:")
    print(solve_1d_fem(4)[2])


# =============================================================================
# 实验5：预条件子（fig3）
# =============================================================================
def run_preconditioner():
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg, spilu, LinearOperator

    print("=" * 60)
    print("实验：预条件子对迭代法收敛的影响")
    print("=" * 60)

    def build_2d_laplacian(n):
        N = n * n
        h = 1.0 / (n + 1)
        main_diag = 4 * np.ones(N) / h ** 2
        side_diag = -np.ones(N - 1) / h ** 2
        for i in range(1, n):
            side_diag[i * n - 1] = 0
        updown_diag = -np.ones(N - n) / h ** 2
        return diags([main_diag, side_diag, side_diag, updown_diag, updown_diag], [0, -1, 1, -n, n], format='csr')

    def cg_with_history(A, b, M_inv=None, tol=1e-10, maxiter=500):
        residuals = [np.linalg.norm(b)]

        def callback(xk):
            residuals.append(np.linalg.norm(b - A @ xk))
        x0 = np.zeros_like(b)
        kw = dict(x0=x0, rtol=tol, maxiter=maxiter, callback=callback)
        x, info = cg(A, b, M=M_inv, **kw) if M_inv is not None else cg(A, b, **kw)
        return x, residuals, info

    n_grid = 30
    A = build_2d_laplacian(n_grid)
    N = n_grid * n_grid
    eigvals = np.linalg.eigvalsh(A.toarray())
    kappa_original = eigvals.max() / eigvals.min()
    print(f"矩阵大小 {N}×{N}, 条件数 κ(A) = {kappa_original:.1f}")
    b = np.ones(N)
    _, res1, _ = cg_with_history(A, b, None)
    M_jacobi_inv = diags(1.0 / A.diagonal(), format='csr')
    _, res2, _ = cg_with_history(A, b, M_jacobi_inv)
    ilu = spilu(A.tocsc())
    _, res3, _ = cg_with_history(A, b, LinearOperator((N, N), matvec=ilu.solve))
    print(f"迭代次数: 无预条件 {len(res1) - 1}, Jacobi {len(res2) - 1}, ILU {len(res3) - 1}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    ax6 = axes3[0]
    ax6.semilogy(res1, 'b-', linewidth=2, label=f'无预条件 ({len(res1) - 1}次)')
    ax6.semilogy(res2, 'g-', linewidth=2, label=f'Jacobi ({len(res2) - 1}次)')
    ax6.semilogy(res3, 'r-', linewidth=2, label=f'ILU ({len(res3) - 1}次)')
    ax6.set_xlabel('迭代次数', fontsize=12)
    ax6.set_ylabel('残差 ||b - Ax||', fontsize=12, labelpad=10)
    ax6.set_title(f'预条件子效果对比\n({n_grid}×{n_grid}网格, κ={kappa_original:.0f})', fontsize=14)
    ax6.legend(fontsize=11)
    ax6.set_xlim(0, max(len(res1), len(res2), len(res3)))

    ax7 = axes3[1]
    grid_sizes = [10, 15, 20, 25, 30]
    iters = {'none': [], 'jacobi': [], 'ilu': []}
    for n in grid_sizes:
        A_t = build_2d_laplacian(n)
        b_t = np.ones(n * n)
        iters['none'].append(len(cg_with_history(A_t, b_t, None, maxiter=1000)[1]) - 1)
        iters['jacobi'].append(len(cg_with_history(A_t, b_t, diags(1.0 / A_t.diagonal(), format='csr'), maxiter=1000)[1]) - 1)
        ilu_t = spilu(A_t.tocsc())
        iters['ilu'].append(len(cg_with_history(A_t, b_t, LinearOperator((n * n, n * n), matvec=ilu_t.solve), maxiter=1000)[1]) - 1)
        print(f"  {n}×{n}: 无预条件={iters['none'][-1]}, Jacobi={iters['jacobi'][-1]}, ILU={iters['ilu'][-1]}")
    ax7.plot(grid_sizes, iters['none'], 'bo-', linewidth=2, markersize=8, label='无预条件')
    ax7.plot(grid_sizes, iters['jacobi'], 'gs-', linewidth=2, markersize=8, label='Jacobi')
    ax7.plot(grid_sizes, iters['ilu'], 'r^-', linewidth=2, markersize=8, label='ILU')
    ax7.set_xlabel('网格大小 n', fontsize=12)
    ax7.set_ylabel('迭代次数', fontsize=12, labelpad=10)
    ax7.set_title('网格加密时的迭代次数增长\n(预条件子能有效控制增长)', fontsize=14)
    ax7.legend(fontsize=11)

    ax8 = axes3[2]
    theta = np.linspace(0, 2 * np.pi, 100)
    a1, b1 = 3, 0.3
    ax8.plot(a1 * np.cos(theta), b1 * np.sin(theta), 'b-', linewidth=2, label=f'原问题等高线 (κ={a1 / b1:.0f})')
    a2, b2 = 1.5, 1.2
    ax8.plot(a2 * np.cos(theta), b2 * np.sin(theta), 'r-', linewidth=2, label=f'预条件后 (κ={a2 / b2:.1f})')
    ax8.annotate('', xy=(0, 0), xytext=(2.5, 0.25), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax8.annotate('', xy=(1.5, -0.2), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax8.annotate('', xy=(0, 0), xytext=(1.5, -0.2), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, ls='--'))
    ax8.annotate('', xy=(0, 0), xytext=(1.2, 0.8), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax8.plot(0, 0, 'k*', markersize=15, label='最优解')
    ax8.set_xlim(-4, 4); ax8.set_ylim(-2, 2); ax8.set_aspect('equal')
    ax8.set_title('条件数的几何意义\n(扁椭圆→zigzag, 圆→直达)', fontsize=14)
    ax8.legend(loc='upper right', fontsize=10)
    ax8.set_xlabel('$x_1$', fontsize=12); ax8.set_ylabel('$x_2$', fontsize=12)
    plt.tight_layout()
    fig3.savefig('figs/chap13_fig3.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("预条件子实验图像已保存到 figs/chap13_fig3.png")


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if only in ('all', 'perf'):
        run_performance_experiments()
    if only in ('all', 'fem'):
        run_fem()
    if only in ('all', 'precond'):
        run_preconditioner()
    print("\n第13章代码执行完成！")
