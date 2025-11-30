"""
第2章扩展：Python优化工具包演示
比较scipy.optimize, cvxpy在不同规模问题上的性能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 示例1：线性规划 - 资源分配问题
# =============================================================================
def solve_linear_program_scipy(n_vars, n_constraints):
    """使用scipy.optimize.linprog求解线性规划"""
    np.random.seed(42)
    c = np.random.rand(n_vars)  # 目标函数系数
    A_ub = np.random.rand(n_constraints, n_vars)  # 不等式约束矩阵
    b_ub = np.random.rand(n_constraints) * n_vars  # 约束右端项

    start = time.time()
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    elapsed = time.time() - start

    return result.fun if result.success else None, elapsed

# =============================================================================
# 示例2：二次规划 - 投资组合优化
# =============================================================================
def portfolio_optimization_scipy(n_assets):
    """投资组合优化：最小化风险，约束收益"""
    np.random.seed(42)
    # 生成协方差矩阵（保证正定）
    A = np.random.rand(n_assets, n_assets)
    Q = A @ A.T + np.eye(n_assets) * 0.1  # 协方差矩阵
    r = np.random.rand(n_assets) * 0.1 + 0.05  # 预期收益
    target_return = 0.08

    def objective(w):
        return 0.5 * w @ Q @ w

    def grad(w):
        return Q @ w

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: w @ r - target_return}  # 最低收益
    ]
    bounds = [(0, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    start = time.time()
    result = minimize(objective, w0, method='SLSQP', jac=grad,
                     constraints=constraints, bounds=bounds)
    elapsed = time.time() - start

    return result.fun if result.success else None, elapsed, result.x if result.success else None

# =============================================================================
# 示例3：非凸优化 - Rosenbrock函数
# =============================================================================
def rosenbrock_optimization():
    """非凸优化：Rosenbrock函数"""
    def rosenbrock(x):
        return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

    results = {}
    methods = ['Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell']
    dims = [2, 5, 10, 20]

    for dim in dims:
        results[dim] = {}
        x0 = np.zeros(dim)
        for method in methods:
            start = time.time()
            try:
                res = minimize(rosenbrock, x0, method=method,
                             options={'maxiter': 10000})
                elapsed = time.time() - start
                results[dim][method] = {
                    'value': res.fun,
                    'time': elapsed,
                    'success': res.success,
                    'nfev': res.nfev
                }
            except:
                results[dim][method] = {'value': None, 'time': None, 'success': False}

    return results

# =============================================================================
# 性能测试
# =============================================================================
def benchmark_linear_programming():
    """线性规划规模性能测试"""
    sizes = [10, 50, 100, 200, 500, 1000]
    times = []

    for n in sizes:
        _, t = solve_linear_program_scipy(n, n//2)
        times.append(t)
        print(f"LP: n={n}, time={t:.4f}s")

    return sizes, times

def benchmark_quadratic_programming():
    """二次规划规模性能测试"""
    sizes = [10, 20, 50, 100, 200, 500]
    times = []
    risks = []

    for n in sizes:
        risk, t, _ = portfolio_optimization_scipy(n)
        times.append(t)
        risks.append(risk)
        print(f"QP: n={n}, time={t:.4f}s, risk={risk:.6f}")

    return sizes, times, risks

# =============================================================================
# 生成图表
# =============================================================================
def plot_lp_performance(sizes, times, save_path):
    """绘制线性规划性能图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(sizes, times, 'bo-', markersize=8, linewidth=2, label='scipy.linprog (HiGHS)')
    ax.set_xlabel('Problem Size (variables)', fontsize=12)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Linear Programming: Performance vs Problem Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # 添加注释
    ax.annotate(f'{times[-1]:.3f}s', xy=(sizes[-1], times[-1]),
                xytext=(sizes[-1]*0.7, times[-1]*2),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_qp_performance(sizes, times, risks, save_path):
    """绘制二次规划性能图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 计算时间
    ax1.semilogy(sizes, times, 'ro-', markersize=8, linewidth=2)
    ax1.set_xlabel('Number of Assets', fontsize=12)
    ax1.set_ylabel('Computation Time (s)', fontsize=12)
    ax1.set_title('Portfolio Optimization: Time Complexity', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 最优风险
    ax2.plot(sizes, risks, 'gs-', markersize=8, linewidth=2)
    ax2.set_xlabel('Number of Assets', fontsize=12)
    ax2.set_ylabel('Minimum Portfolio Risk', fontsize=12)
    ax2.set_title('Optimal Risk vs Number of Assets', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_nonconvex_comparison(results, save_path):
    """绘制非凸优化方法比较图"""
    methods = ['Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell']
    dims = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 计算时间比较
    width = 0.2
    x = np.arange(len(dims))

    for i, method in enumerate(methods):
        times = [results[d][method]['time'] if results[d][method]['time'] else 0 for d in dims]
        ax1.bar(x + i*width, times, width, label=method, alpha=0.8)

    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Rosenbrock: Optimization Time by Method', fontsize=14)
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 函数调用次数比较
    for i, method in enumerate(methods):
        nfev = [results[d][method].get('nfev', 0) if results[d][method]['success'] else 0 for d in dims]
        ax2.bar(x + i*width, nfev, width, label=method, alpha=0.8)

    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Function Evaluations', fontsize=12)
    ax2.set_title('Rosenbrock: Function Evaluations by Method', fontsize=14)
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels([str(d) for d in dims])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_portfolio_weights(save_path):
    """绘制投资组合权重分配图"""
    _, _, weights = portfolio_optimization_scipy(10)

    fig, ax = plt.subplots(figsize=(8, 5))

    assets = [f'Asset {i+1}' for i in range(len(weights))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))

    bars = ax.bar(assets, weights, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Optimal Portfolio Allocation (10 Assets)', fontsize=14)
    ax.set_ylim(0, max(weights)*1.2)

    # 在柱子上标注数值
    for bar, w in zip(bars, weights):
        if w > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{w:.2f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Equal weight (0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_algorithm_comparison(save_path):
    """绘制算法适用性比较表格"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    data = [
        ['scipy.linprog', 'LP', 'HiGHS', '< 0.1s @ 1000var', 'Fast, reliable'],
        ['scipy.minimize\n(SLSQP)', 'QP/NLP', 'SQP', '< 1s @ 500var', 'General purpose'],
        ['scipy.minimize\n(L-BFGS-B)', 'Unconstrained', 'Quasi-Newton', '< 0.5s @ 100dim', 'Large-scale'],
        ['cvxpy + ECOS', 'Convex', 'Interior Point', '< 1s @ 1000var', 'Modeling friendly'],
        ['cvxpy + MOSEK', 'SDP/SOCP', 'Interior Point', '< 1s @ 500var', 'Commercial solver'],
    ]

    columns = ['Tool', 'Problem Type', 'Algorithm', 'Typical Speed', 'Remarks']

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                    cellLoc='center', colWidths=[0.18, 0.15, 0.15, 0.22, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置交替行颜色
    for i in range(1, len(data)+1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')

    ax.set_title('Python Optimization Tools Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap02"

    print("=" * 60)
    print("Running optimization benchmarks...")
    print("=" * 60)

    # 1. 线性规划性能测试
    print("\n[1] Linear Programming Benchmark")
    lp_sizes, lp_times = benchmark_linear_programming()
    plot_lp_performance(lp_sizes, lp_times, f"{output_dir}/lp_performance.pdf")

    # 2. 二次规划性能测试
    print("\n[2] Quadratic Programming Benchmark")
    qp_sizes, qp_times, qp_risks = benchmark_quadratic_programming()
    plot_qp_performance(qp_sizes, qp_times, qp_risks, f"{output_dir}/qp_performance.pdf")

    # 3. 投资组合权重
    print("\n[3] Portfolio Weights")
    plot_portfolio_weights(f"{output_dir}/portfolio_weights.pdf")

    # 4. 非凸优化比较
    print("\n[4] Non-convex Optimization Comparison")
    rosenbrock_results = rosenbrock_optimization()
    plot_nonconvex_comparison(rosenbrock_results, f"{output_dir}/nonconvex_comparison.pdf")

    # 5. 算法比较表
    print("\n[5] Algorithm Comparison Table")
    plot_algorithm_comparison(f"{output_dir}/algorithm_comparison.pdf")

    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)
