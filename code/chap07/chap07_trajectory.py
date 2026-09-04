"""
第7章：轨迹优化 - 可视化演示

包含：
1. 最省力移动问题：解析解可视化
2. MPC滚动时域控制演示
3. 直接配点法与打靶法对比
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 导入中文字体配置
import sys
sys.path.insert(0, 'code')
from plot_utils import setup_chinese_font
setup_chinese_font()


def create_minimum_effort_figure():
    """
    最省力移动问题：从x=0移动到x=1，最小化控制能量

    min ∫₀ᵀ u²(t) dt
    s.t. x'' = u, x(0)=0, x'(0)=0, x(T)=1, x'(T)=0

    解析解：u*(t) = 6/T² * (1 - 2t/T)
           x*(t) = (t/T)² * (3 - 2t/T)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T = 1.0  # 总时间
    t = np.linspace(0, T, 100)

    # 解析解
    u_star = 6 / T**2 * (1 - 2*t/T)
    x_star = (t/T)**2 * (3 - 2*t/T)
    v_star = 6*t/T**2 * (1 - t/T)  # x'(t) = 6t/T² - 6t²/T³

    # === 子图1: 最优轨迹 ===
    ax1 = axes[0]
    ax1.plot(t, x_star, 'b-', linewidth=2.5, label='位置 x(t)')
    ax1.plot(t, v_star, 'g--', linewidth=2, label='速度 v(t)')

    ax1.scatter([0, T], [0, 1], color='red', s=100, zorder=5)
    ax1.annotate('起点 (0,0)', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=10)
    ax1.annotate('终点 (T,1)', (T, 1), textcoords="offset points", xytext=(-50, 10), fontsize=10)

    ax1.set_xlabel('时间 t', fontsize=12)
    ax1.set_ylabel('x, v', fontsize=12)
    ax1.set_title('最优轨迹 (T=1)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 子图2: 最优控制 ===
    ax2 = axes[1]
    ax2.plot(t, u_star, 'r-', linewidth=2.5)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.fill_between(t, 0, u_star, alpha=0.3, color='red')

    ax2.set_xlabel('时间 t', fontsize=12)
    ax2.set_ylabel('控制 u(t)', fontsize=12)
    ax2.set_title('最优控制 u*(t) = 6/T²(1-2t/T)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标注控制的物理意义
    ax2.annotate('加速\n(u > 0)', (0.15, u_star[15]),
                 textcoords="offset points", xytext=(20, 20), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('减速\n(u < 0)', (0.85, u_star[85]),
                 textcoords="offset points", xytext=(-60, -30), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # === 子图3: 不同T值的能量消耗 ===
    ax3 = axes[2]

    T_values = np.linspace(0.5, 3.0, 50)
    energy = []

    for T_val in T_values:
        t_val = np.linspace(0, T_val, 100)
        u_val = 6 / T_val**2 * (1 - 2*t_val/T_val)
        # 能量 = ∫u²dt
        E = np.trapz(u_val**2, t_val)
        energy.append(E)

    ax3.plot(T_values, energy, 'b-', linewidth=2.5)
    ax3.set_xlabel('时间范围 T', fontsize=12)
    ax3.set_ylabel('总能量 ∫u²dt', fontsize=12)
    ax3.set_title('能量与时间: 越慢越省力', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 标注几个点
    for T_mark in [0.5, 1.0, 2.0]:
        idx = np.argmin(np.abs(T_values - T_mark))
        ax3.scatter(T_values[idx], energy[idx], s=80, zorder=5)
        ax3.annotate(f'T={T_mark}: E={energy[idx]:.1f}', (T_values[idx], energy[idx]),
                     textcoords="offset points", xytext=(10, 10), fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap07_fig1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap07_fig1.png")


def create_mpc_demo():
    """
    MPC（模型预测控制）滚动时域演示

    系统：一阶积分器 x' = u
    目标：从x=0到达x=1并停留
    约束：|u| <= 0.5
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 系统参数
    dt = 0.1  # 时间步长
    N = 10    # 预测时域
    x_target = 1.0
    u_max = 0.5

    # MPC求解函数
    def mpc_solve(x_current, N, x_target, dt, u_max):
        """求解MPC优化问题"""
        def objective(u_seq):
            x = x_current
            cost = 0
            for u in u_seq:
                x = x + dt * u
                cost += (x - x_target)**2 + 0.1 * u**2
            return cost

        # 初始猜测
        u0 = np.zeros(N)

        # 优化
        bounds = [(-u_max, u_max)] * N
        result = minimize(objective, u0, method='SLSQP', bounds=bounds)
        return result.x

    # 模拟MPC控制
    T_sim = 50  # 模拟步数
    x_history = [0.0]
    u_history = []
    planned_trajectories = []  # 存储每步的预测轨迹

    x = 0.0
    for step in range(T_sim):
        # 求解MPC
        u_seq = mpc_solve(x, N, x_target, dt, u_max)

        # 存储预测轨迹
        x_pred = [x]
        for u in u_seq:
            x_pred.append(x_pred[-1] + dt * u)
        planned_trajectories.append((step * dt, x_pred))

        # 应用第一个控制
        u = u_seq[0]
        u_history.append(u)

        # 系统演化
        x = x + dt * u
        x_history.append(x)

    t_sim = np.arange(len(x_history)) * dt

    # === 子图1: 状态轨迹 ===
    ax1 = axes[0, 0]
    ax1.plot(t_sim, x_history, 'b-', linewidth=2, label='实际轨迹')
    ax1.axhline(y=x_target, color='r', linestyle='--', label=f'目标 x={x_target}')

    # 画几条预测轨迹
    for step in [0, 5, 15, 30]:
        if step < len(planned_trajectories):
            t_start, x_pred = planned_trajectories[step]
            t_pred = t_start + np.arange(len(x_pred)) * dt
            ax1.plot(t_pred, x_pred, 'g--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('状态 x', fontsize=12)
    ax1.set_title('MPC: 滚动时域控制', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 子图2: 控制输入 ===
    ax2 = axes[0, 1]
    t_u = np.arange(len(u_history)) * dt
    ax2.step(t_u, u_history, 'r-', linewidth=2, where='post', label='控制 u')
    ax2.axhline(y=u_max, color='gray', linestyle='--', label=f'u_max={u_max}')
    ax2.axhline(y=-u_max, color='gray', linestyle='--')
    ax2.fill_between(t_u, -u_max, u_max, alpha=0.1, color='gray')

    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('控制 u', fontsize=12)
    ax2.set_title('MPC: 控制输入 (带饱和约束)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === 子图3: 滚动时域示意 ===
    ax3 = axes[1, 0]

    # 画三个时刻的预测时域
    for step, color in zip([0, 10, 25], ['blue', 'green', 'orange']):
        if step < len(planned_trajectories):
            t_start, x_pred = planned_trajectories[step]
            t_pred = t_start + np.arange(len(x_pred)) * dt

            # 预测轨迹
            ax3.plot(t_pred, x_pred, '-', color=color, linewidth=2,
                     label=f't={t_start:.1f}秒: 预测时域')

            # 标记当前时刻
            ax3.scatter(t_start, x_pred[0], s=100, color=color, zorder=5)

            # 标记预测时域范围
            ax3.axvspan(t_start, t_start + N*dt, alpha=0.1, color=color)

    ax3.axhline(y=x_target, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('时间 (秒)', fontsize=12)
    ax3.set_ylabel('状态 x', fontsize=12)
    ax3.set_title('滚动时域: 每步更新计划', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 4)

    # === 子图4: 跟踪误差 ===
    ax4 = axes[1, 1]
    error = np.array(x_history) - x_target
    ax4.plot(t_sim, error, 'b-', linewidth=2)
    ax4.axhline(y=0, color='gray', linestyle='--')
    ax4.fill_between(t_sim, 0, error, alpha=0.3)

    ax4.set_xlabel('时间 (秒)', fontsize=12)
    ax4.set_ylabel('跟踪误差 (x - 目标)', fontsize=12)
    ax4.set_title('MPC: 跟踪误差', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap07_fig2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap07_fig2.png")


def create_shooting_vs_collocation():
    """
    对比直接配点法和打靶法
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 简单的边值问题：y'' = -y, y(0)=0, y(π)=0
    # 解析解：y = sin(x)

    x_span = np.linspace(0, np.pi, 100)
    y_exact = np.sin(x_span)

    # === 子图1: 打靶法示意 ===
    ax1 = axes[0]

    def shoot(y0_prime):
        """从y(0)=0, y'(0)=y0_prime出发积分"""
        sol = solve_ivp(lambda t, y: [y[1], -y[0]], (0, np.pi), [0, y0_prime],
                        t_eval=x_span)
        return sol.y[0]

    # 几个不同的初始斜率
    for y0p, color in zip([0.5, 1.0, 1.5, 2.0], ['gray', 'gray', 'gray', 'gray']):
        y_shot = shoot(y0p)
        ax1.plot(x_span, y_shot, '--', color=color, alpha=0.5, linewidth=1)
        ax1.annotate(f"y'(0)={y0p}", (0.1, y0p*0.1), fontsize=9, color=color)

    # 正确的解
    y_correct = shoot(1.0)
    ax1.plot(x_span, y_correct, 'b-', linewidth=2.5, label='正确的射击')
    ax1.plot(x_span, y_exact, 'r--', linewidth=2, label='解析解')

    ax1.scatter([0, np.pi], [0, 0], color='red', s=100, zorder=5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('打靶法: 调整 y\'(0)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 子图2: 配点法示意 ===
    ax2 = axes[1]

    # 配点位置
    n_colloc = 5
    x_colloc = np.linspace(0, np.pi, n_colloc + 2)  # 包含边界

    ax2.plot(x_span, y_exact, 'b-', linewidth=2, label='解析解')
    ax2.scatter(x_colloc[1:-1], np.sin(x_colloc[1:-1]), s=150, c='red',
                marker='o', zorder=5, label='配点')
    ax2.scatter(x_colloc[[0, -1]], [0, 0], s=150, c='green',
                marker='s', zorder=5, label='边界条件')

    # 画连接线表示多项式
    for x_c in x_colloc[1:-1]:
        ax2.axvline(x=x_c, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('配点法: 在配点处强制满足ODE', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === 子图3: 方法对比 ===
    ax3 = axes[2]

    methods = ['打靶法', '配点法', '直接转录']
    pros = [
        '实现简单\n适合类初值问题',
        '稀疏雅可比\n适合刚性系统',
        '最灵活\n可处理路径约束'
    ]
    cons = [
        '对初值敏感',
        '某些基函数\n矩阵稠密',
        'NLP规模大'
    ]

    # 创建表格形式的对比
    table_data = []
    for i, (method, pro, con) in enumerate(zip(methods, pros, cons)):
        table_data.append([method, pro, con])

    ax3.axis('off')
    table = ax3.table(
        cellText=table_data,
        colLabels=['方法', '优点', '缺点'],
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.35, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('轨迹优化方法对比', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap07_fig3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap07_fig3.png")


if __name__ == "__main__":
    print("=== 第7章：轨迹优化可视化 ===\n")

    print("1. 生成最省力移动问题图...")
    create_minimum_effort_figure()

    print("\n2. 生成MPC演示图...")
    create_mpc_demo()

    print("\n3. 生成打靶法vs配点法对比图...")
    create_shooting_vs_collocation()

    print("\n所有图像生成完成!")
