#!/usr/bin/env python3
"""
第13章：有限元与性能实验演示

这里演示书中提到的几个关键实验：
1. Python GIL 对多线程的影响
2. 按行访问 vs 按列访问的速度差异
3. NumPy vs 纯Python的性能对比
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from multiprocessing import Pool

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第13章：性能实验演示")
print("="*60)

# ==================== 实验1：GIL的影响 ====================
print("\n" + "="*60)
print("实验1：Python GIL 对多线程的影响")
print("="*60)

def count_small():
    """一个简单的计数任务(缩小规模)"""
    n = 0
    for _ in range(5_000_000):  # 缩小10倍以节省时间
        n += 1
    return n

# 单线程测试
print("\n测试单线程...")
t0 = time.time()
count_small()
single_time = time.time() - t0
print(f"单线程耗时: {single_time:.2f} 秒")

# 双线程测试
print("\n测试双线程...")
t0 = time.time()
t1 = threading.Thread(target=count_small)
t2 = threading.Thread(target=count_small)
t1.start()
t2.start()
t1.join()
t2.join()
double_time = time.time() - t0
print(f"双线程耗时: {double_time:.2f} 秒")

print(f"\n结论：双线程并没有更快，反而可能更慢！")
print(f"这是因为Python的GIL（全局解释器锁）使得多线程无法真正并行执行计算密集型任务。")

# ==================== 实验2：按行访问 vs 按列访问 ====================
print("\n" + "="*60)
print("实验2：按行访问 vs 按列访问（Cache效应）")
print("="*60)

N = 2000  # 缩小规模，原书是5000
A = np.random.rand(N, N)

# 按行访问
print("\n测试按行访问...")
t0 = time.time()
s = 0.0
for i in range(N):
    for j in range(N):
        s += A[i, j]
row_time = time.time() - t0
print(f"按行访问: {row_time:.2f} 秒")

# 按列访问
print("\n测试按列访问...")
t0 = time.time()
s = 0.0
for j in range(N):
    for i in range(N):
        s += A[i, j]
col_time = time.time() - t0
print(f"按列访问: {col_time:.2f} 秒")

print(f"\n速度比: 按列/按行 = {col_time/row_time:.2f}x")
print("结论：按列访问比按行访问慢很多，因为NumPy默认按行存储（row-major）。")
print("按行访问时CPU可以利用Cache局部性，按列访问时频繁Cache miss。")

# ==================== 实验3：NumPy vs 纯Python ====================
print("\n" + "="*60)
print("实验3：NumPy vs 纯Python")
print("="*60)

size = 1_000_000  # 缩小规模，原书是5百万
x = np.random.rand(size)
y = np.random.rand(size)

# 纯Python
print("\n测试纯 Python...")
t0 = time.time()
z1 = [x[i] + y[i] for i in range(size)]
python_time = time.time() - t0
print(f"纯 Python: {python_time:.3f} 秒")

# NumPy
print("\n测试 NumPy...")
t0 = time.time()
z2 = x + y
numpy_time = time.time() - t0
print(f"NumPy: {numpy_time:.3f} 秒")

speedup = python_time / numpy_time
print(f"\nNumPy 比纯 Python 快了 {speedup:.1f} 倍！")

# ==================== 可视化结果 ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1: GIL实验结果
ax1 = axes[0]
bars1 = ax1.bar(['单线程', '双线程'],
                [single_time, double_time],
                color=['steelblue', 'coral'])
ax1.set_ylabel('时间 (秒)', labelpad=10)
ax1.set_title('实验1: Python GIL效应\n(多线程不能加速CPU密集型任务)')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s', ha='center', va='bottom')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 图2: Cache实验结果
ax2 = axes[1]
bars2 = ax2.bar(['按行访问(快)', '按列访问(慢)'],
                [row_time, col_time],
                color=['green', 'red'])
ax2.set_ylabel('时间 (秒)', labelpad=10)
ax2.set_title('实验2: 内存访问模式\n(按行访问利用Cache局部性更快)')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s', ha='center', va='bottom')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 图3: NumPy vs Python结果
ax3 = axes[2]
bars3 = ax3.bar(['纯Python', 'NumPy'],
                [python_time, numpy_time],
                color=['orange', 'blue'])
ax3.set_ylabel('时间 (秒)', labelpad=10)
ax3.set_title(f'实验3: NumPy vs 纯Python\n(NumPy快{speedup:.0f}倍!)')
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}s', ha='center', va='bottom')
ax3.set_yscale('log')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figs/chap13_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap13_fig1.png")

# ==================== 有限元1D热传导求解演示 ====================
print("\n" + "="*60)
print("有限元方法：1D热传导问题")
print("="*60)

def solve_1d_fem(n_elements, f_source=1.0):
    """
    用有限元法求解1D热传导问题：
    -u''(x) = f(x), x in [0,1]
    u(0) = u(1) = 0

    使用线性单元
    """
    n_nodes = n_elements + 1
    h = 1.0 / n_elements  # 单元长度

    # 刚度矩阵组装
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # 单元刚度矩阵 (线性单元)
    # K_e = (1/h) * [[1, -1], [-1, 1]]
    ke = (1/h) * np.array([[1, -1], [-1, 1]])

    # 单元载荷向量 (假设f=常数)
    fe = f_source * h / 2 * np.array([1, 1])

    # 组装全局矩阵
    for e in range(n_elements):
        i, j = e, e + 1
        K[i, i] += ke[0, 0]
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[j, j] += ke[1, 1]
        F[i] += fe[0]
        F[j] += fe[1]

    # 应用边界条件 u(0) = u(1) = 0
    K_reduced = K[1:-1, 1:-1]
    F_reduced = F[1:-1]

    # 求解线性系统
    u_inner = np.linalg.solve(K_reduced, F_reduced)

    # 完整解向量
    u = np.zeros(n_nodes)
    u[1:-1] = u_inner

    x = np.linspace(0, 1, n_nodes)

    return x, u, K, F

# 解析解 (f=1 的情况): u(x) = x(1-x)/2
def exact_solution(x):
    return x * (1 - x) / 2

# 不同网格精度的求解
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 有限元解与解析解对比
ax4 = axes2[0]
x_fine = np.linspace(0, 1, 200)
ax4.plot(x_fine, exact_solution(x_fine), 'k-', linewidth=2, label='解析解: $u = x(1-x)/2$')

colors = ['blue', 'green', 'orange', 'red']
for i, n_elem in enumerate([4, 8, 16, 32]):
    x_fem, u_fem, _, _ = solve_1d_fem(n_elem)
    ax4.plot(x_fem, u_fem, 'o-', color=colors[i], markersize=4,
             label=f'{n_elem}个单元')

ax4.set_xlabel('x', labelpad=10)
ax4.set_ylabel('u(x)', labelpad=10)
ax4.set_title('一维有限元: 热传导问题\n$-u\'\' = 1$, $u(0) = u(1) = 0$')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 图2: 误差收敛
ax5 = axes2[1]
n_elements_list = [4, 8, 16, 32, 64, 128]
errors = []

for n_elem in n_elements_list:
    x_fem, u_fem, _, _ = solve_1d_fem(n_elem)
    u_exact = exact_solution(x_fem)
    error = np.max(np.abs(u_fem - u_exact))
    errors.append(error)

h_list = [1/n for n in n_elements_list]

ax5.loglog(h_list, errors, 'bo-', linewidth=2, markersize=8, label='有限元误差')
# 添加二阶收敛参考线
h_ref = np.array(h_list)
ax5.loglog(h_ref, 0.5 * h_ref**2, 'r--', linewidth=1.5, label='$O(h^2)$参考线')

ax5.set_xlabel('单元尺寸 h', labelpad=10)
ax5.set_ylabel('最大误差', labelpad=10)
ax5.set_title('有限元收敛速率\n(线性单元为二阶收敛)')
ax5.legend()
ax5.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('figs/chap13_fig2.png', dpi=150, bbox_inches='tight')
print("图像已保存到 figs/chap13_fig2.png")

# 打印刚度矩阵结构（小规模）
print("\n有限元刚度矩阵示例（4个单元）：")
x, u, K, F = solve_1d_fem(4)
print("刚度矩阵 K (5x5):")
print(K)
print(f"\n载荷向量 F: {F}")
print(f"\n解向量 u: {u}")
print(f"\n解析解（节点处）: {exact_solution(x)}")
print(f"最大误差: {np.max(np.abs(u - exact_solution(x))):.6f}")

# ==================== 实验4：预条件子效果演示 ====================
print("\n" + "="*60)
print("实验4：预条件子对迭代法收敛的影响")
print("="*60)

print("""
预条件子(Preconditioner)的核心思想：
- 迭代法（如共轭梯度法）的收敛速度取决于矩阵的"条件数" κ
- 条件数 = 最大特征值 / 最小特征值
- κ 越大，收敛越慢；κ 接近1时，一步就能收敛

预条件子 M 的作用：把原问题 Ax = b 变换成 M⁻¹Ax = M⁻¹b
如果 M 选得好，M⁻¹A 的条件数会比 A 小很多！
""")

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator

def build_2d_laplacian(n):
    """
    构建 2D 拉普拉斯矩阵（有限元/有限差分的典型矩阵）
    这个矩阵代表了 -∇²u = f 的离散化

    n: 每个方向的网格点数
    返回: n²×n² 的稀疏矩阵
    """
    N = n * n  # 总自由度
    h = 1.0 / (n + 1)  # 网格步长

    # 主对角线：4/h²
    main_diag = 4 * np.ones(N) / h**2

    # 相邻点（左右）：-1/h²
    side_diag = -np.ones(N - 1) / h**2
    # 每行末尾不应该连接到下一行开头
    for i in range(1, n):
        side_diag[i * n - 1] = 0

    # 上下相邻点：-1/h²
    updown_diag = -np.ones(N - n) / h**2

    A = diags([main_diag, side_diag, side_diag, updown_diag, updown_diag],
              [0, -1, 1, -n, n], format='csr')
    return A

def cg_with_history(A, b, M_inv=None, tol=1e-10, maxiter=500):
    """
    带历史记录的共轭梯度法
    返回解和每次迭代的残差
    """
    residuals = []

    def callback(xk):
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))

    x0 = np.zeros_like(b)
    residuals.append(np.linalg.norm(b))  # 初始残差

    # 兼容不同版本的 scipy (tol -> rtol)
    if M_inv is not None:
        x, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M_inv, callback=callback)
    else:
        x, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, callback=callback)

    return x, residuals, info

# 测试不同网格大小
print("\n构建测试矩阵...")
n_grid = 30  # 30×30 网格 → 900×900 矩阵
A = build_2d_laplacian(n_grid)
N = n_grid * n_grid

print(f"矩阵大小: {N}×{N}")
print(f"非零元素: {A.nnz} (稀疏度: {100*A.nnz/N**2:.2f}%)")

# 计算条件数（只对小矩阵可行）
A_dense = A.toarray()
eigvals = np.linalg.eigvalsh(A_dense)
kappa_original = eigvals.max() / eigvals.min()
print(f"原始条件数 κ(A) = {kappa_original:.1f}")

# 构造右端向量
b = np.ones(N)

# ===== 测试1：无预条件 =====
print("\n1. 无预条件的 CG...")
x1, res1, info1 = cg_with_history(A, b, M_inv=None)
print(f"   迭代次数: {len(res1)-1}, 收敛状态: {'成功' if info1==0 else '未完全收敛'}")

# ===== 测试2：Jacobi预条件 =====
print("\n2. Jacobi 预条件 (M = diag(A))...")
# Jacobi预条件：M = 对角线元素
M_jacobi_diag = A.diagonal()
M_jacobi_inv = diags(1.0 / M_jacobi_diag, format='csr')
x2, res2, info2 = cg_with_history(A, b, M_inv=M_jacobi_inv)
print(f"   迭代次数: {len(res2)-1}, 收敛状态: {'成功' if info2==0 else '未完全收敛'}")

# ===== 测试3：ILU预条件 =====
print("\n3. ILU 预条件 (不完全LU分解)...")
# ILU: 近似的LU分解，保持稀疏性
ilu = spilu(A.tocsc())
M_ilu_inv = LinearOperator((N, N), matvec=ilu.solve)
x3, res3, info3 = cg_with_history(A, b, M_inv=M_ilu_inv)
print(f"   迭代次数: {len(res3)-1}, 收敛状态: {'成功' if info3==0 else '未完全收敛'}")

# ===== 可视化 =====
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

# 图1：收敛曲线对比
ax6 = axes3[0]
ax6.semilogy(res1, 'b-', linewidth=2, label=f'无预条件 ({len(res1)-1}次)')
ax6.semilogy(res2, 'g-', linewidth=2, label=f'Jacobi ({len(res2)-1}次)')
ax6.semilogy(res3, 'r-', linewidth=2, label=f'ILU ({len(res3)-1}次)')
ax6.set_xlabel('迭代次数', fontsize=12)
ax6.set_ylabel('残差 ||b - Ax||', fontsize=12, labelpad=10)
ax6.set_title(f'预条件子效果对比\n({n_grid}×{n_grid}网格, κ={kappa_original:.0f})', fontsize=14)
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, max(len(res1), len(res2), len(res3)))

# 图2：条件数与迭代次数的关系
ax7 = axes3[1]
grid_sizes = [10, 15, 20, 25, 30]
iters_no_precond = []
iters_jacobi = []
iters_ilu = []
kappas = []

print("\n测试不同网格大小的迭代次数...")
for n in grid_sizes:
    A_test = build_2d_laplacian(n)
    b_test = np.ones(n*n)

    # 条件数
    if n <= 30:
        eigv = np.linalg.eigvalsh(A_test.toarray())
        kappas.append(eigv.max() / eigv.min())

    # 无预条件
    _, r1, _ = cg_with_history(A_test, b_test, None, maxiter=1000)
    iters_no_precond.append(len(r1)-1)

    # Jacobi
    M_j = diags(1.0 / A_test.diagonal(), format='csr')
    _, r2, _ = cg_with_history(A_test, b_test, M_j, maxiter=1000)
    iters_jacobi.append(len(r2)-1)

    # ILU
    ilu_test = spilu(A_test.tocsc())
    M_i = LinearOperator((n*n, n*n), matvec=ilu_test.solve)
    _, r3, _ = cg_with_history(A_test, b_test, M_i, maxiter=1000)
    iters_ilu.append(len(r3)-1)

    print(f"  {n}×{n}: 无预条件={iters_no_precond[-1]}, Jacobi={iters_jacobi[-1]}, ILU={iters_ilu[-1]}")

ax7.plot(grid_sizes, iters_no_precond, 'bo-', linewidth=2, markersize=8, label='无预条件')
ax7.plot(grid_sizes, iters_jacobi, 'gs-', linewidth=2, markersize=8, label='Jacobi')
ax7.plot(grid_sizes, iters_ilu, 'r^-', linewidth=2, markersize=8, label='ILU')
ax7.set_xlabel('网格大小 n', fontsize=12)
ax7.set_ylabel('迭代次数', fontsize=12, labelpad=10)
ax7.set_title('网格加密时的迭代次数增长\n(预条件子能有效控制增长)', fontsize=14)
ax7.legend(fontsize=11)
ax7.grid(True, alpha=0.3)

# 图3：直观解释 - 条件数的几何意义
ax8 = axes3[2]

# 画两个椭圆：一个扁（条件数大），一个圆（条件数小）
theta = np.linspace(0, 2*np.pi, 100)

# 原始问题（椭圆很扁）
a1, b1 = 3, 0.3  # 条件数 = 10
x_ellipse1 = a1 * np.cos(theta)
y_ellipse1 = b1 * np.sin(theta)
ax8.plot(x_ellipse1, y_ellipse1, 'b-', linewidth=2, label=f'原问题等高线 (κ={a1/b1:.0f})')

# 预条件后（椭圆接近圆）
a2, b2 = 1.5, 1.2  # 条件数 ≈ 1.25
x_ellipse2 = a2 * np.cos(theta)
y_ellipse2 = b2 * np.sin(theta)
ax8.plot(x_ellipse2, y_ellipse2, 'r-', linewidth=2, label=f'预条件后 (κ={a2/b2:.1f})')

# 画迭代路径示意
# 扁椭圆的zigzag路径
ax8.annotate('', xy=(0, 0), xytext=(2.5, 0.25),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax8.annotate('', xy=(1.5, -0.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax8.annotate('', xy=(0, 0), xytext=(1.5, -0.2),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, ls='--'))

# 圆的直线路径
ax8.annotate('', xy=(0, 0), xytext=(1.2, 0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax8.plot(0, 0, 'k*', markersize=15, label='最优解')
ax8.set_xlim(-4, 4)
ax8.set_ylim(-2, 2)
ax8.set_aspect('equal')
ax8.set_title('条件数的几何意义\n(扁椭圆→zigzag, 圆→直达)', fontsize=14)
ax8.legend(loc='upper right', fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.set_xlabel('$x_1$', fontsize=12)
ax8.set_ylabel('$x_2$', fontsize=12)

plt.tight_layout()
plt.savefig('figs/chap13_fig3.png', dpi=150, bbox_inches='tight')
print("\n预条件子实验图像已保存到 figs/chap13_fig3.png")

# 打印总结
print("\n" + "="*60)
print("预条件子效果总结")
print("="*60)
print(f"""
关键发现：
1. 原始条件数 κ = {kappa_original:.0f}（2D拉普拉斯矩阵，{n_grid}×{n_grid}网格）

2. 迭代次数对比（30×30网格）：
   - 无预条件: {len(res1)-1} 次
   - Jacobi:   {len(res2)-1} 次 (加速 {(len(res1)-1)/(len(res2)-1):.1f}×)
   - ILU:      {len(res3)-1} 次 (加速 {(len(res1)-1)/(len(res3)-1):.1f}×)

3. 理论解释：
   - CG法的迭代次数 ≈ O(√κ)
   - 对于2D有限元，κ = O(h⁻²) = O(n²)
   - 所以无预条件时，迭代次数 ≈ O(n)（线性增长）
   - 好的预条件子能让 κ 保持为 O(1)，迭代次数不随网格增长！

4. 实际意义：
   - 工业级有限元可能有 n = 1000（百万自由度）
   - 无预条件需要 ~1000 次迭代
   - 多重网格预条件只需 ~10 次迭代
   - 这就是 100倍 的性能差距！
""")

plt.show()
print("\n第13章代码执行完成！")
