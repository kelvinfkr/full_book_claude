#!/usr/bin/env python3
"""
第1章：KKT系统求解演示

本代码演示如何用KKT条件求解约束优化问题，包括：
1. 将约束优化问题转化为KKT线性系统
2. 求解KKT系统得到原始变量(x,y,z)和对偶变量(λ)
3. 可视化解的几何意义
4. 理解拉格朗日乘子的"影子价格"含义

问题设置：
    min  f(x,y,z) = x² + y² + z²   （最小化到原点的距离平方）
    s.t. g(x,y,z) = x + y + z - 1 = 0  （点必须在平面上）

拉格朗日函数：
    L(x,y,z,λ) = x² + y² + z² + λ(x + y + z - 1)

KKT条件（一阶必要条件）：
    ∂L/∂x = 2x + λ = 0
    ∂L/∂y = 2y + λ = 0
    ∂L/∂z = 2z + λ = 0
    ∂L/∂λ = x + y + z - 1 = 0

对应书中：图1.1-1.3
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入中文字体配置（确保matplotlib能正确显示中文）
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

# ==================== 第一部分：构建并求解KKT系统 ====================
print("="*60)
print("第1章：KKT系统求解演示")
print("问题：求原点到平面 x + y + z = 1 的最短距离")
print("="*60)

# KKT系统的矩阵形式：K @ [x, y, z, λ]^T = b
#
# 把4个KKT条件写成线性系统：
# [2  0  0  1] [x]   [0]    <- ∂L/∂x = 2x + λ = 0
# [0  2  0  1] [y] = [0]    <- ∂L/∂y = 2y + λ = 0
# [0  0  2  1] [z]   [0]    <- ∂L/∂z = 2z + λ = 0
# [1  1  1  0] [λ]   [1]    <- 约束条件 x + y + z = 1
#
# 矩阵K的结构：
# - 左上角3×3块：目标函数的Hessian矩阵 ∇²f = 2I
# - 右上角3×1块：约束梯度 ∇g = [1,1,1]^T
# - 左下角1×3块：约束梯度的转置 (∇g)^T
# - 右下角1×1块：0（因为没有不等式约束）

K = np.array([
    [2, 0, 0, 1],  # ∂L/∂x = 0 方程
    [0, 2, 0, 1],  # ∂L/∂y = 0 方程
    [0, 0, 2, 1],  # ∂L/∂z = 0 方程
    [1, 1, 1, 0]   # 约束条件 g = 0
], dtype=float)

# 右端向量：前三个是0（∇f的相反数），最后一个是约束右端值
b = np.array([0, 0, 0, 1], dtype=float)

print("\nKKT矩阵 K:")
print(K)
print("\n右端向量 b:", b)

# 使用numpy求解线性系统 K @ solution = b
# 这一步就是"同时求解原始问题和对偶问题"
solution = np.linalg.solve(K, b)

# 解向量包含4个分量：3个原始变量 + 1个对偶变量
x, y, z, lam = solution

print("\n解向量:")
print(f"  x = {x:.6f}")  # 最优点的x坐标
print(f"  y = {y:.6f}")  # 最优点的y坐标
print(f"  z = {z:.6f}")  # 最优点的z坐标
print(f"  λ = {lam:.6f}")  # 拉格朗日乘子（影子价格）

# ==================== 第二部分：验证解的正确性 ====================

# 手算结果对比：由对称性，最优点应该是 (1/3, 1/3, 1/3)
# 代入 2x + λ = 0 得 λ = -2/3
print("\n手算结果: (1/3, 1/3, 1/3, -2/3)")
print(f"数值结果: ({x:.6f}, {y:.6f}, {z:.6f}, {lam:.6f})")

# 验证解是否满足约束条件
constraint = x + y + z
print(f"\n验证约束 x + y + z = {constraint:.6f} (应该等于 1)")

# 计算原点到最优点的距离（这就是最终答案）
distance = np.sqrt(x**2 + y**2 + z**2)
print(f"\n原点到平面的最短距离: {distance:.6f}")
print(f"理论值: 1/√3 = {1/np.sqrt(3):.6f}")

# ==================== 第三部分：可视化 ====================
# 创建三个子图展示不同方面
fig = plt.figure(figsize=(14, 5))

# --- 图1：3D空间中的几何可视化 ---
ax1 = fig.add_subplot(131, projection='3d')

# 画约束平面 x + y + z = 1
# 用网格方法：固定x,y，计算z = 1 - x - y
xx, yy = np.meshgrid(
    np.linspace(-0.2, 0.8, 20),  # x的范围
    np.linspace(-0.2, 0.8, 20)   # y的范围
)
zz = 1 - xx - yy  # 由平面方程解出z

# 只显示z在合理范围内的部分
mask = (zz >= -0.2) & (zz <= 0.8)
zz_masked = np.where(mask, zz, np.nan)

# 画半透明的蓝色平面
ax1.plot_surface(xx, yy, zz_masked, alpha=0.3, color='blue', label='平面 x+y+z=1')

# 画原点（红色大圆点）
ax1.scatter([0], [0], [0], color='red', s=100, marker='o', label='原点')

# 画最近点（绿色星号）
ax1.scatter([x], [y], [z], color='green', s=100, marker='*', label='最近点')

# 画从原点到最近点的连线（这就是最短距离）
ax1.plot([0, x], [0, y], [0, z], 'g-', linewidth=2, label=f'距离 = {distance:.4f}')

# 设置坐标轴标签和标题
ax1.set_xlabel('X', labelpad=10)
ax1.set_ylabel('Y', labelpad=10)
ax1.set_zlabel('Z')
ax1.set_title('三维可视化：点到平面的距离')
ax1.legend(fontsize=8)

# --- 图2：KKT矩阵的结构可视化 ---
ax2 = fig.add_subplot(132)

# 变量名标签
labels = ['x', 'y', 'z', 'λ']

# 用热力图显示KKT矩阵
im = ax2.imshow(K, cmap='RdBu_r', aspect='auto')

# 设置刻度标签
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(['∂L/∂x=0', '∂L/∂y=0', '∂L/∂z=0', '约束条件'])

# 在每个格子中添加数值
for i in range(4):
    for j in range(4):
        # 根据背景色选择文字颜色（深色背景用白字）
        text_color = 'white' if abs(K[i,j]) > 0.5 else 'black'
        ax2.text(j, i, f'{K[i,j]:.0f}', ha='center', va='center',
                fontsize=12, color=text_color)

ax2.set_title('KKT矩阵结构')
plt.colorbar(im, ax=ax2)

# --- 图3：解向量的可视化 ---
ax3 = fig.add_subplot(133)

# 变量名和对应的数值
variables = ['x*', 'y*', 'z*', 'λ*']
values = [x, y, z, lam]
# 原始变量用蓝色，对偶变量用橙色
colors = ['blue', 'blue', 'blue', 'orange']

# 画柱状图
bars = ax3.bar(variables, values, color=colors, edgecolor='black')

# 在每个柱子上方添加数值标注
for bar, val in zip(bars, values):
    height = bar.get_height()
    # 正数标在上方，负数标在下方
    va = 'bottom' if height > 0 else 'top'
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}', ha='center', va=va, fontsize=11)

# 画y=0的参考线
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('数值', labelpad=10)
ax3.set_title('KKT系统解\n(蓝色: 原始变量, 橙色: 对偶变量)')
ax3.set_ylim(-1, 0.5)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figs/chap01_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap01_fig1.png")

# ==================== 第四部分：影子价格的经济学解释 ====================
print("\n" + "="*60)
print("影子价格解释")
print("="*60)

# 核心问题：如果约束从 x+y+z=1 变成 x+y+z=1+ε，最优值会如何变化？
# 理论上：df*/dε = -λ（这就是影子价格的含义）
#
# 物理解释：
# - λ < 0 意味着约束"拉"住了目标函数
# - |λ| 越大，约束越"值钱"——放松约束能带来更多改进

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 图4：最优值随约束松弛量的变化 ---

# 扫描不同的约束右端值（ε从-0.3到0.3）
epsilons = np.linspace(-0.3, 0.3, 50)
distances = []  # 存储每个ε对应的最优目标值

for eps in epsilons:
    # 修改约束右端值：x + y + z = 1 + ε
    b_new = np.array([0, 0, 0, 1 + eps], dtype=float)
    # 重新求解KKT系统
    sol_new = np.linalg.solve(K, b_new)
    x_n, y_n, z_n, _ = sol_new
    # 计算目标函数值（距离的平方）
    dist = np.sqrt(x_n**2 + y_n**2 + z_n**2)
    distances.append(dist**2)  # 注意：我们优化的是距离的平方

ax4 = axes[0]

# 画最优值曲线
ax4.plot(epsilons, distances, 'b-', linewidth=2)

# 标注原始解（ε=0时）
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.scatter([0], [distance**2], color='red', s=100, zorder=5,
           label=f'原始解: f*={distance**2:.4f}')

# 画切线来验证 df*/dε = -λ
# 在ε=0处，切线斜率应该等于 -λ
slope = -lam  # 理论上的斜率
tangent = distance**2 + slope * epsilons  # 切线方程
ax4.plot(epsilons, tangent, 'r--', linewidth=1.5,
        label=f'切线 (斜率 = -λ = {-lam:.4f})')

ax4.set_xlabel('约束松弛量 ε', labelpad=10)
ax4.set_ylabel('最优目标值 f*(ε)', labelpad=10)
ax4.set_title('影子价格: λ = df*/dε\n(最优值对约束的敏感度)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- 图5：最优点随约束变化的轨迹 ---
ax5 = axes[1]

# 选取几个代表性的ε值
eps_vals = [-0.2, -0.1, 0, 0.1, 0.2]
# 用不同颜色区分
colors_list = plt.cm.viridis(np.linspace(0, 1, len(eps_vals)))

for i, eps in enumerate(eps_vals):
    # 修改约束并求解
    b_new = np.array([0, 0, 0, 1 + eps], dtype=float)
    sol_new = np.linalg.solve(K, b_new)
    x_n, y_n, z_n, _ = sol_new

    # 画出最优解（由对称性，只画x*，因为x*=y*=z*）
    ax5.scatter([1+eps], [x_n], color=colors_list[i], s=100, marker='o')
    # 添加标注
    ax5.annotate(f'ε={eps:.1f}', (1+eps, x_n),
                textcoords="offset points", xytext=(5,5), fontsize=9)

ax5.set_xlabel('约束右端项 (1 + ε)', labelpad=10)
ax5.set_ylabel('最优解 x* (= y* = z*)', labelpad=10)
ax5.set_title('最优解随约束变化')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/chap01_fig2.png', dpi=150, bbox_inches='tight')
print("图像已保存到 figs/chap01_fig2.png")

plt.show()
print("\n第1章代码执行完成！")
