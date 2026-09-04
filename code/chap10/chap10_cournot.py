#!/usr/bin/env python3
"""
第10章：多智能体与博弈均衡演示

演示Cournot双寡头竞争的纳什均衡求解
"""
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体配置
import sys
sys.path.insert(0, 'code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第10章：多智能体与博弈均衡演示")
print("="*60)

# ==================== Cournot竞争模型 ====================
print("\n" + "="*60)
print("Cournot双寡头竞争模型")
print("="*60)

class CournotGame:
    """
    Cournot双寡头竞争模型

    参数说明：
    - a: 市场需求截距（需求曲线P = a - b*Q中的a）
    - b: 市场需求斜率（价格对产量的敏感度）
    - c: 边际成本（生产每单位产品的成本）

    利润函数：
    pi_i(q_i, q_j) = (a - b*(q_i + q_j)) * q_i - c * q_i
    """

    def __init__(self, a=100, b=1, c=10):
        self.a = a
        self.b = b
        self.c = c

    def profit(self, q1, q2):
        """计算两家公司的利润"""
        Q = q1 + q2  # 总产量
        P = self.a - self.b * Q  # 市场价格

        pi1 = P * q1 - self.c * q1
        pi2 = P * q2 - self.c * q2

        return pi1, pi2

    def best_response(self, q_other):
        """
        计算最优响应

        数学推导：
        pi(q, q_other) = (a - b*(q + q_other)) * q - c * q
        d(pi)/d(q) = (a - c - b*q_other) - 2*b*q = 0
        解得：q* = (a - c - b*q_other) / (2*b)
        """
        q_star = (self.a - self.c - self.b * q_other) / (2 * self.b)
        return max(0, q_star)

    def nash_equilibrium_analytical(self):
        """
        解析求解纳什均衡

        由对称性：q1* = q2* = q*
        q* = (a - c - b*q*) / (2*b)
        3*b*q* = a - c
        q* = (a - c) / (3*b)
        """
        q_nash = (self.a - self.c) / (3 * self.b)
        pi_nash = self.b * q_nash ** 2
        return q_nash, pi_nash


def best_response_iteration(game, q1_init=0, q2_init=0, max_iters=100, tol=1e-6):
    """最优响应迭代求解纳什均衡"""
    q1, q2 = q1_init, q2_init
    history = [(q1, q2)]

    print(f"初始: q1={q1:.2f}, q2={q2:.2f}")

    for iteration in range(max_iters):
        q1_new = game.best_response(q2)
        q2_new = game.best_response(q1_new)

        history.append((q1_new, q2_new))

        if abs(q1_new - q1) < tol and abs(q2_new - q2) < tol:
            print(f"在第{iteration+1}次迭代后收敛")
            break

        if iteration < 5:
            pi1, pi2 = game.profit(q1_new, q2_new)
            print(f"迭代{iteration+1}: q1={q1_new:.4f}, q2={q2_new:.4f}, 利润=({pi1:.2f}, {pi2:.2f})")

        q1, q2 = q1_new, q2_new

    return q1, q2, history


# 创建游戏
game = CournotGame(a=100, b=1, c=10)

print(f"\n参数: a={game.a}, b={game.b}, c={game.c}")
print(f"价格函数: P = {game.a} - {game.b}*(q1+q2)")
print(f"成本函数: C_i = {game.c}*q_i")

# 解析解
q_nash, pi_nash = game.nash_equilibrium_analytical()
print(f"\n解析解:")
print(f"  纳什均衡产量: q* = {q_nash:.4f}")
print(f"  纳什均衡利润: pi* = {pi_nash:.4f}")

# 迭代解
print(f"\n最优响应迭代:")
q1, q2, history = best_response_iteration(game, q1_init=0, q2_init=0)

print(f"\n最终结果:")
print(f"  q1 = {q1:.4f}, q2 = {q2:.4f}")
pi1, pi2 = game.profit(q1, q2)
print(f"  利润: pi1 = {pi1:.2f}, pi2 = {pi2:.2f}")

# 验证纳什均衡
print("\n验证纳什均衡（单方面改变不会更好）:")
print("如果公司1偏离:")
for factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
    q1_dev = q1 * factor
    pi1_dev, _ = game.profit(q1_dev, q2)
    change = "更好" if pi1_dev > pi1 else "更差或相同"
    print(f"  q1={q1_dev:.2f} -> 利润={pi1_dev:.2f} ({change})")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：最优响应曲线
ax1 = axes[0, 0]
q_range = np.linspace(0, 50, 100)
br1 = [game.best_response(q) for q in q_range]  # 公司1的最优响应
br2 = [game.best_response(q) for q in q_range]  # 公司2的最优响应（对称）

ax1.plot(q_range, br1, 'b-', linewidth=2, label="公司1最优响应: $q_1^*(q_2)$")
ax1.plot(br2, q_range, 'r-', linewidth=2, label="公司2最优响应: $q_2^*(q_1)$")
ax1.scatter([q_nash], [q_nash], color='green', s=200, zorder=5, marker='*', label=f'纳什均衡 ({q_nash:.1f}, {q_nash:.1f})')

# 绘制迭代轨迹
history_arr = np.array(history)
ax1.plot(history_arr[:, 1], history_arr[:, 0], 'k--', linewidth=1, alpha=0.7)
ax1.scatter(history_arr[:, 1], history_arr[:, 0], color='orange', s=50, zorder=4)

ax1.set_xlabel('$q_2$ (公司2产量)', labelpad=10)
ax1.set_ylabel('$q_1$ (公司1产量)', labelpad=10)
ax1.set_title('最优响应曲线\n(交点 = 纳什均衡)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 50)

# 图2：利润曲面
ax2 = axes[0, 1]
q1_grid = np.linspace(0, 60, 50)
q2_grid = np.linspace(0, 60, 50)
Q1, Q2 = np.meshgrid(q1_grid, q2_grid)
Pi1 = np.zeros_like(Q1)
for i in range(len(q1_grid)):
    for j in range(len(q2_grid)):
        Pi1[i, j], _ = game.profit(Q1[i, j], Q2[i, j])

contour = ax2.contourf(Q1, Q2, Pi1, levels=30, cmap='viridis')
ax2.contour(Q1, Q2, Pi1, levels=10, colors='white', linewidths=0.5, alpha=0.5)
ax2.scatter([q_nash], [q_nash], color='red', s=200, marker='*', zorder=5, label='纳什均衡')
plt.colorbar(contour, ax=ax2, label='利润 $\\pi_1$')
ax2.set_xlabel('$q_1$', labelpad=10)
ax2.set_ylabel('$q_2$', labelpad=10)
ax2.set_title('公司1的利润曲面\n(给定公司2的产量)', fontsize=12)
ax2.legend()

# 图3：迭代收敛
ax3 = axes[1, 0]
iterations = range(len(history))
q1_hist = [h[0] for h in history]
q2_hist = [h[1] for h in history]

ax3.plot(iterations, q1_hist, 'b-o', linewidth=2, markersize=6, label='$q_1$')
ax3.plot(iterations, q2_hist, 'r-s', linewidth=2, markersize=6, label='$q_2$')
ax3.axhline(y=q_nash, color='green', linestyle='--', linewidth=2, label=f'纳什均衡 $q^*={q_nash:.2f}$')

ax3.set_xlabel('迭代次数', labelpad=10)
ax3.set_ylabel('产量', labelpad=10)
ax3.set_title('最优响应迭代收敛过程', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：KKT条件解释
ax4 = axes[1, 1]
ax4.axis('off')

text = """
Cournot竞争：用KKT条件求解纳什均衡

每家公司解决以下优化问题：
    max_{q_i} π_i(q_i, q_{-i}) = (a - b(q_i + q_{-i}))q_i - cq_i
    s.t.  q_i ≥ 0

一阶条件（KKT）：
    ∂π_i/∂q_i = a - c - 2bq_i - bq_{-i} = 0

纳什均衡要求所有公司的KKT条件同时满足：
    a - c - 2bq₁* - bq₂* = 0
    a - c - bq₁* - 2bq₂* = 0

由对称性解得：q₁* = q₂* = (a-c)/(3b) = """ + f"{q_nash:.2f}" + """

关键结论：
• 纳什均衡 = 所有玩家KKT条件的联立解
• 拉格朗日乘子表示约束的"影子价格"
• 最优响应迭代能收敛到纳什均衡

验证：q* = (100-10)/(3×1) = 30 ✓
"""

ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figs/chap10_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap10_fig1.png")

plt.show()
print("\n第10章代码执行完成！")
