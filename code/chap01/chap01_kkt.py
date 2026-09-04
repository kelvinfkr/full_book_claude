#!/usr/bin/env python3
"""
第1章：KKT系统求解演示
求解原点到平面 x + y + z = 1 的最短距离问题
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
figs_dir = os.path.join(project_root, 'figures/chap01')
os.makedirs(figs_dir, exist_ok=True)

print("="*60)
print("第1章：KKT系统求解演示")
print("问题：求原点到平面 x + y + z = 1 的最短距离")
print("="*60)

# KKT系统矩阵
# [2  0  0  1] [x]   [0]
# [0  2  0  1] [y] = [0]
# [0  0  2  1] [z]   [0]
# [1  1  1  0] [λ]   [1]

K = np.array([
    [2, 0, 0, 1],
    [0, 2, 0, 1],
    [0, 0, 2, 1],
    [1, 1, 1, 0]
], dtype=float)

b = np.array([0, 0, 0, 1], dtype=float)

print("\nKKT矩阵 K:")
print(K)
print("\n右端向量 b:", b)

# 求解线性系统
solution = np.linalg.solve(K, b)
x, y, z, lam = solution

print("\n解向量:")
print(f"  x = {x:.6f}")
print(f"  y = {y:.6f}")
print(f"  z = {z:.6f}")
print(f"  λ = {lam:.6f}")

# 手算结果对比
print("\n手算结果: (1/3, 1/3, 1/3, -2/3)")
print(f"数值结果: ({x:.6f}, {y:.6f}, {z:.6f}, {lam:.6f})")

# 验证约束
constraint = x + y + z
print(f"\n验证约束 x + y + z = {constraint:.6f} (应该等于 1)")

# 计算最短距离
distance = np.sqrt(x**2 + y**2 + z**2)
print(f"\n原点到平面的最短距离: {distance:.6f}")
print(f"理论值: 1/√3 = {1/np.sqrt(3):.6f}")

# ==================== 可视化 ====================
fig = plt.figure(figsize=(14, 5))

# 图1：3D可视化
ax1 = fig.add_subplot(131, projection='3d')

# 画平面 x + y + z = 1
xx, yy = np.meshgrid(np.linspace(-0.2, 0.8, 20), np.linspace(-0.2, 0.8, 20))
zz = 1 - xx - yy
# 只显示有效区域
mask = (zz >= -0.2) & (zz <= 0.8)
zz_masked = np.where(mask, zz, np.nan)
ax1.plot_surface(xx, yy, zz_masked, alpha=0.3, color='blue', label='平面 x+y+z=1')

# 画原点
ax1.scatter([0], [0], [0], color='red', s=100, marker='o', label='原点')

# 画最近点
ax1.scatter([x], [y], [z], color='green', s=100, marker='*', label='最近点')

# 画连线（最短距离）
ax1.plot([0, x], [0, y], [0, z], 'g-', linewidth=2, label=f'距离 = {distance:.4f}')

ax1.set_xlabel('X', labelpad=10)
ax1.set_ylabel('Y', labelpad=10)
ax1.set_zlabel('Z')
ax1.set_title('三维可视化：点到平面的距离')
ax1.legend(fontsize=8)

# 图2：KKT矩阵结构热力图
ax2 = fig.add_subplot(132)
labels = ['x', 'y', 'z', 'λ']
im = ax2.imshow(K, cmap='RdBu_r', aspect='auto')
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(['∂L/∂x=0', '∂L/∂y=0', '∂L/∂z=0', '约束条件'])

# 添加数值标注
for i in range(4):
    for j in range(4):
        ax2.text(j, i, f'{K[i,j]:.0f}', ha='center', va='center', fontsize=12,
                color='white' if abs(K[i,j]) > 0.5 else 'black')

ax2.set_title('KKT矩阵结构')
plt.colorbar(im, ax=ax2)

# 图3：解的可视化
ax3 = fig.add_subplot(133)
variables = ['x*', 'y*', 'z*', 'λ*']
values = [x, y, z, lam]
colors = ['blue', 'blue', 'blue', 'orange']
bars = ax3.bar(variables, values, color=colors, edgecolor='black')

# 添加数值标注
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=11)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('数值', labelpad=10)
ax3.set_title('KKT系统解\n(蓝色: 原始变量, 橙色: 对偶变量)')
ax3.set_ylim(-1, 0.5)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
save_path1 = os.path.join(figs_dir, 'kkt_visualization.pdf')
plt.savefig(save_path1, dpi=150, bbox_inches='tight')
print(f"\n图像已保存到 {save_path1}")

# ==================== 拉格朗日乘子的影子价格解释 ====================
print("\n" + "="*60)
print("影子价格解释")
print("="*60)

# 如果约束从 x+y+z=1 变成 x+y+z=1+ε，最优值如何变化？
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# 扫描不同的约束右端值
epsilons = np.linspace(-0.3, 0.3, 50)
distances = []
for eps in epsilons:
    b_new = np.array([0, 0, 0, 1 + eps], dtype=float)
    sol_new = np.linalg.solve(K, b_new)
    x_n, y_n, z_n, _ = sol_new
    dist = np.sqrt(x_n**2 + y_n**2 + z_n**2)
    distances.append(dist**2)  # 我们优化的是距离的平方

ax4 = axes[0]
ax4.plot(epsilons, distances, 'b-', linewidth=2)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.scatter([0], [distance**2], color='red', s=100, zorder=5, label=f'原始解: f*={distance**2:.4f}')

# 画切线（斜率=λ）
slope = -lam  # df*/dε = -λ
tangent = distance**2 + slope * epsilons
ax4.plot(epsilons, tangent, 'r--', linewidth=1.5, label=f'切线 (斜率 = -λ = {-lam:.4f})')

ax4.set_xlabel('约束松弛量 ε', labelpad=10)
ax4.set_ylabel('最优目标值 f*(ε)', labelpad=10)
ax4.set_title('影子价格: λ = df*/dε\n(最优值对约束的敏感度)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 图2：约束逐渐放松时最优点的变化
ax5 = axes[1]
eps_vals = [-0.2, -0.1, 0, 0.1, 0.2]
colors_list = plt.cm.viridis(np.linspace(0, 1, len(eps_vals)))

for i, eps in enumerate(eps_vals):
    b_new = np.array([0, 0, 0, 1 + eps], dtype=float)
    sol_new = np.linalg.solve(K, b_new)
    x_n, y_n, z_n, _ = sol_new
    ax5.scatter([1+eps], [x_n], color=colors_list[i], s=100, marker='o')
    ax5.annotate(f'ε={eps:.1f}', (1+eps, x_n), textcoords="offset points", xytext=(5,5), fontsize=9)

ax5.set_xlabel('约束右端项 (1 + ε)', labelpad=10)
ax5.set_ylabel('最优解 x* (= y* = z*)', labelpad=10)
ax5.set_title('最优解随约束变化')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
save_path2 = os.path.join(figs_dir, 'shadow_price.pdf')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"图像已保存到 {save_path2}")

print("\n第1章代码执行完成！")
