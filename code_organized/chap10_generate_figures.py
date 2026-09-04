"""
生成chap15所需的所有图表
多智能体系统：从微观博弈到宏观平均场

所有图表使用中文标注
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# ==============================================================================
# 图1: 囚徒困境可视化
# ==============================================================================
def plot_prisoners_dilemma():
    """可视化囚徒困境的支付矩阵和纳什均衡"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 支付矩阵热力图
    ax = axes[0]
    payoff_p1 = np.array([[1, 3], [0, 2]])
    payoff_p2 = np.array([[1, 0], [3, 2]])

    # 绘制两个矩阵叠加
    im = ax.imshow(payoff_p1, cmap='RdYlGn_r', alpha=0.7, vmin=0, vmax=3)

    # 添加数值标注
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'({payoff_p1[i,j]}, {payoff_p2[i,j]})',
                   ha='center', va='center', fontsize=14, fontweight='bold')

    # 标记纳什均衡
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False,
                               edgecolor='red', linewidth=3, linestyle='--'))
    ax.text(1, 1.7, '纳什均衡', ha='center', color='red', fontsize=11)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['合作', '背叛'])
    ax.set_yticklabels(['合作', '背叛'])
    ax.set_xlabel('玩家2', fontsize=12)
    ax.set_ylabel('玩家1', fontsize=12)
    ax.set_title("囚徒困境支付矩阵\n(玩家1代价, 玩家2代价)", fontsize=12)

    # 右图: 最优响应分析
    ax = axes[1]
    strategies = ['合作', '背叛']

    # 玩家1的最优响应
    x = np.arange(2)
    width = 0.35

    # 当玩家2合作时，玩家1的代价
    p2_coop = [1, 0]
    # 当玩家2背叛时，玩家1的代价
    p2_defect = [3, 2]

    bars1 = ax.bar(x - width/2, p2_coop, width, label='玩家2选合作', color='steelblue')
    bars2 = ax.bar(x + width/2, p2_defect, width, label='玩家2选背叛', color='coral')

    # 标记最优响应
    ax.annotate('更优!', xy=(1 - width/2, 0.1), fontsize=10, ha='center', color='green')
    ax.annotate('更优!', xy=(1 + width/2, 2.1), fontsize=10, ha='center', color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(['合作', '背叛'])
    ax.set_xlabel("玩家1的策略", fontsize=12)
    ax.set_ylabel("玩家1的代价（刑期/年）", fontsize=12)
    ax.set_title("玩家1的最优响应分析\n（无论对手如何选，背叛总是更优）", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 4)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/prisoners_dilemma.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/prisoners_dilemma.png', bbox_inches='tight')
    plt.close()
    print("Generated: prisoners_dilemma.pdf/png")


# ==============================================================================
# 图2: 交通均衡 (Braess悖论) - 经典例子
# ==============================================================================
def solve_braess_equilibrium(with_braess_edge=True):
    """
    求解经典Braess网络的均衡

    网络结构:
        S ----(t=x/100)---- A ----(t=45)---- E
         \                     \  (t=0)   /
          \                     \        /
           \                     \      /
            ----(t=45)------- B ----(t=x/100)----

    需求: 4000辆车从S到E

    分析:
    - 无Braess边时: 对称均衡，每条路径2000辆，时间=20+45=65分钟
    - 有Braess边时: 由于免费边的存在，所有人都选择S->A->B->E，时间=40+0+40=80分钟
    """
    demand = 4000

    if with_braess_edge:
        f = [0, 0, 4000]  # 所有人走S->A->B->E
        x_SA = 4000
        x_BE = 4000
        path_times = [40 + 45, 45 + 40, 40 + 0 + 40]  # [85, 85, 80]
        avg_time = 80.0
        return f, path_times, avg_time * demand, avg_time
    else:
        f = [2000, 2000]
        path_times = [65, 65]
        avg_time = 65.0
        return f, path_times, avg_time * demand, avg_time


def plot_braess_paradox():
    """可视化Braess悖论"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 求解两种情况
    f_with, t_with, total_with, avg_with = solve_braess_equilibrium(with_braess_edge=True)
    f_without, t_without, total_without, avg_without = solve_braess_equilibrium(with_braess_edge=False)

    # 左图: 有Braess边的网络
    ax = axes[0]
    nodes = {'S': (0, 1), 'A': (1.5, 2), 'B': (1.5, 0), 'E': (3, 1)}

    node_labels = {'S': '起点S', 'A': 'A', 'B': 'B', 'E': '终点E'}
    for name, pos in nodes.items():
        color = 'lightcoral' if name in ['S', 'E'] else 'lightblue'
        ax.plot(*pos, 'o', markersize=30, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=14, fontweight='bold')

    # 边信息 (有Braess边时所有人走S->A->B->E)
    edges = [
        ('S', 'A', 4000, 't=x/100\n流量=4000'),
        ('A', 'E', 0, 't=45\n(未使用)'),
        ('S', 'B', 0, 't=45\n(未使用)'),
        ('B', 'E', 4000, 't=x/100\n流量=4000'),
        ('A', 'B', 4000, 't=0 (免费!)\n流量=4000'),
    ]

    for start, end, flow, label in edges:
        p1, p2 = nodes[start], nodes[end]
        if '免费' in label:
            color = 'red'
            lw = 4
        elif '未使用' in label:
            color = 'gray'
            lw = 1
        else:
            color = 'steelblue'
            lw = max(1, flow / 1000)
        ax.annotate('', xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        if start == 'A' and end == 'B':
            offset = (0.3, 0)
        else:
            offset = (0.15, 0.15)
        ax.text(mid[0]+offset[0], mid[1]+offset[1], label, fontsize=7, ha='center')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'有免费道路\n平均出行时间: {avg_with:.0f}分钟', fontsize=11)

    # 中图: 无Braess边的网络
    ax = axes[1]

    for name, pos in nodes.items():
        color = 'lightgreen' if name in ['S', 'E'] else 'lightblue'
        ax.plot(*pos, 'o', markersize=30, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=14, fontweight='bold')

    edges_no = [
        ('S', 'A', f_without[0], f't=x/100\n流量={f_without[0]:.0f}'),
        ('A', 'E', f_without[0], f't=45\n流量={f_without[0]:.0f}'),
        ('S', 'B', f_without[1], f't=45\n流量={f_without[1]:.0f}'),
        ('B', 'E', f_without[1], f't=x/100\n流量={f_without[1]:.0f}'),
    ]

    for start, end, flow, label in edges_no:
        p1, p2 = nodes[start], nodes[end]
        lw = max(1, flow / 1000)
        ax.annotate('', xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle='->', color='seagreen', lw=lw))
        mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        ax.text(mid[0]+0.15, mid[1]+0.15, label, fontsize=7, ha='center')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'无免费道路\n平均出行时间: {avg_without:.0f}分钟', fontsize=11)

    # 右图: 对比
    ax = axes[2]
    scenarios = ['有免费道路', '无免费道路']
    times = [avg_with, avg_without]
    colors = ['coral', 'seagreen']

    bars = ax.bar(scenarios, times, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('平均出行时间（分钟）', fontsize=12)
    ax.set_title('Braess悖论:\n增加免费道路反而更慢!', fontsize=11)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{t:.0f}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylim(0, max(times) * 1.3)
    ax.grid(axis='y', alpha=0.3)

    # 改进百分比
    if avg_with > avg_without:
        improvement = (avg_with - avg_without) / avg_with * 100
        ax.text(0.5, 0.05, f'移除免费道路可减少{improvement:.1f}%的出行时间!',
               transform=ax.transAxes, ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/braess_paradox.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/braess_paradox.png', bbox_inches='tight')
    plt.close()
    print("Generated: braess_paradox.pdf/png")

    return avg_with, avg_without


# ==============================================================================
# 图3: 图博弈与迭代最优响应
# ==============================================================================
def plot_network_game():
    """可视化图博弈和迭代最优响应收敛"""
    np.random.seed(42)

    # 创建一个小型图 (6个节点)
    n = 6
    # 邻接矩阵 (环形 + 一些额外边)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i+1) % n] = 1
        A[(i+1) % n, i] = 1
    A[0, 3] = A[3, 0] = 1  # 额外边

    # 理想状态
    a0 = np.array([1.0, 0.5, 1.5, 0.8, 1.2, 0.6])

    # 参数
    delta = 0.15  # 策略替代强度

    # 迭代最优响应
    def best_response(a, a0, A, delta):
        """计算最优响应: a_i = a0_i - delta * sum_j A_ij * a_j"""
        return a0 - delta * (A @ a)

    # 运行迭代
    max_iters = 30
    a = np.zeros(n)  # 初始策略
    history = [a.copy()]

    for _ in range(max_iters):
        a = best_response(a, a0, A, delta)
        history.append(a.copy())

    history = np.array(history)

    # 解析解
    a_star = np.linalg.solve(np.eye(n) + delta * A, a0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 左图: 网络结构
    ax = axes[0]
    # 节点位置 (正六边形)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    # 绘制边
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] > 0:
                p1, p2 = pos[i], pos[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.5, linewidth=2)

    # 绘制节点
    for i in range(n):
        color = plt.cm.viridis(a_star[i] / max(a_star))
        ax.plot(*pos[i], 'o', markersize=35, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.text(pos[i][0], pos[i][1], f'{i+1}\n$a^*$={a_star[i]:.2f}',
               ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('图博弈: 6个玩家\n（边=交互关系）', fontsize=11)

    # 中图: 收敛过程
    ax = axes[1]
    for i in range(n):
        ax.plot(history[:, i], label=f'玩家{i+1}', linewidth=2)
        ax.axhline(y=a_star[i], color=f'C{i}', linestyle='--', alpha=0.5)

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('策略 $a_i$', fontsize=12)
    ax.set_title('迭代最优响应收敛过程\n（虚线=纳什均衡）', fontsize=11)
    ax.legend(loc='right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_iters)

    # 右图: 收敛误差
    ax = axes[2]
    errors = [np.linalg.norm(h - a_star) for h in history]
    ax.semilogy(errors, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('误差 $||a^{(k)} - a^*||$', fontsize=12)
    ax.set_title('收敛速率\n（线性收敛）', fontsize=11)
    ax.grid(alpha=0.3)

    # 计算收敛率
    if len(errors) > 5:
        rate = errors[-1] / errors[-2] if errors[-2] > 0 else 0
        ax.text(0.5, 0.9, f'收敛率: {rate:.3f}',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/network_game.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/network_game.png', bbox_inches='tight')
    plt.close()
    print("Generated: network_game.pdf/png")


# ==============================================================================
# 图4: 平均场博弈 (MFG) 数值解
# ==============================================================================
def plot_mfg_solution():
    """可视化平均场博弈的数值解"""
    # 简化的MFG: 线性二次情形的解析解
    nx, nt = 100, 50
    x = np.linspace(-3, 3, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    # 近似解 (高斯分布演化)
    sigma_0 = 0.5
    sigma = lambda s: np.sqrt(sigma_0**2 + 0.1 * s)
    m = np.zeros((nt, nx))
    for i, ti in enumerate(t):
        s = sigma(ti)
        m[i] = np.exp(-x**2 / (2 * s**2)) / (s * np.sqrt(2*np.pi))

    # 价值函数: 二次近似
    u = np.zeros((nt, nx))
    for i, ti in enumerate(t):
        remaining = 1 - ti
        u[i] = 0.5 * x**2 * np.exp(-remaining) + 0.3 * remaining

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 左上: 密度演化 m(x, t)
    ax = axes[0, 0]
    c = ax.contourf(X, T, m, levels=20, cmap='viridis')
    plt.colorbar(c, ax=ax, label='密度 $m(x,t)$')
    ax.set_xlabel('位置 $x$', fontsize=12)
    ax.set_ylabel('时间 $t$', fontsize=12)
    ax.set_title('群体密度演化 $m(x,t)$\n（Fokker-Planck方程）', fontsize=11)

    # 右上: 价值函数 u(x, t)
    ax = axes[0, 1]
    c = ax.contourf(X, T, u, levels=20, cmap='coolwarm')
    plt.colorbar(c, ax=ax, label='价值 $u(x,t)$')
    ax.set_xlabel('位置 $x$', fontsize=12)
    ax.set_ylabel('时间 $t$', fontsize=12)
    ax.set_title('价值函数 $u(x,t)$\n（Hamilton-Jacobi-Bellman方程）', fontsize=11)

    # 左下: 不同时刻的密度剖面
    ax = axes[1, 0]
    times_to_plot = [0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(times_to_plot)))

    for ti, c in zip(times_to_plot, colors):
        idx = int(ti * (nt-1))
        ax.plot(x, m[idx], color=c, linewidth=2, label=f't={ti:.2f}')

    ax.set_xlabel('位置 $x$', fontsize=12)
    ax.set_ylabel('密度 $m(x,t)$', fontsize=12)
    ax.set_title('不同时刻的密度剖面\n（群体逐渐扩散）', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    # 右下: 最优控制 (速度场)
    ax = axes[1, 1]
    # 最优控制 alpha = -u_x
    u_x = np.gradient(u, x[1]-x[0], axis=1)
    velocity = -u_x

    # 绘制向量场
    skip = 5
    ax.quiver(X[::skip, ::skip], T[::skip, ::skip],
             velocity[::skip, ::skip], np.zeros_like(velocity[::skip, ::skip]),
             scale=20, alpha=0.7)

    # 叠加密度轮廓
    ax.contour(X, T, m, levels=5, colors='red', alpha=0.5)

    ax.set_xlabel('位置 $x$', fontsize=12)
    ax.set_ylabel('时间 $t$', fontsize=12)
    ax.set_title('最优速度场 $\\alpha^* = -\\nabla u$\n（箭头表示最优移动方向）', fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/mfg_solution.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/mfg_solution.png', bbox_inches='tight')
    plt.close()
    print("Generated: mfg_solution.pdf/png")


# ==============================================================================
# 图5: QMIX架构示意图 (简化版)
# ==============================================================================
def plot_qmix_architecture():
    """绘制QMIX架构的示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 定义组件位置
    agent_y = 6
    agent_x = [1, 3, 5]
    mix_y = 3
    mix_x = 3
    out_y = 0.5
    out_x = 3
    state_y = 3
    state_x = 7

    # 绘制智能体Q网络
    for i, x in enumerate(agent_x):
        rect = plt.Rectangle((x-0.5, agent_y-0.5), 1, 1,
                             facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, agent_y, f'$Q_{i+1}(o_{i+1}, a_{i+1})$',
               ha='center', va='center', fontsize=10)
        ax.text(x, agent_y+1, f'智能体{i+1}\n局部观测', ha='center', fontsize=9)

    # 绘制混合网络
    rect = plt.Rectangle((mix_x-1.5, mix_y-0.75), 3, 1.5,
                         facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(mix_x, mix_y, '混合网络\n$Q_{tot} = f(Q_1, Q_2, Q_3; s)$',
           ha='center', va='center', fontsize=10)

    # 绘制超网络和全局状态
    rect = plt.Rectangle((state_x-0.8, state_y-0.5), 1.6, 1,
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(state_x, state_y, '全局\n状态 $s$', ha='center', va='center', fontsize=10)

    # 绘制输出
    circle = plt.Circle((out_x, out_y), 0.5, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(out_x, out_y, '$Q_{tot}$', ha='center', va='center', fontsize=12, fontweight='bold')

    # 绘制连接箭头
    arrow_style = dict(arrowstyle='->', color='black', lw=2)

    # Q网络到混合网络
    for x in agent_x:
        ax.annotate('', xy=(mix_x, mix_y+0.75), xytext=(x, agent_y-0.5),
                   arrowprops=arrow_style)

    # 混合网络到输出
    ax.annotate('', xy=(out_x, out_y+0.5), xytext=(mix_x, mix_y-0.75),
               arrowprops=arrow_style)

    # 全局状态到混合网络 (超网络)
    ax.annotate('', xy=(mix_x+1.5, mix_y), xytext=(state_x-0.8, state_y),
               arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--'))
    ax.text(5.5, mix_y+0.8, '超网络\n生成权重', fontsize=9, color='green', ha='center')

    # 添加关键约束说明
    ax.text(0.5, 1.5, '关键约束:\n$\\frac{\\partial Q_{tot}}{\\partial Q_i} \\geq 0$\n（单调性）',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.text(6.5, 1, '保证:\nIGM原则\n（分散执行\n一致性）',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('QMIX架构: 多智能体RL的价值分解方法', fontsize=14)

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/qmix_architecture.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/qmix_architecture.png', bbox_inches='tight')
    plt.close()
    print("Generated: qmix_architecture.pdf/png")


# ==============================================================================
# 图6: 势博弈与收敛性
# ==============================================================================
def plot_potential_game():
    """可视化势博弈的势函数和收敛性质"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 简单的2玩家势博弈
    a1 = np.linspace(-0.5, 2.5, 100)
    a2 = np.linspace(-0.5, 2.5, 100)
    A1, A2 = np.meshgrid(a1, a2)

    Phi = 0.5*(A1 - 1)**2 + 0.5*(A2 - 1)**2 + 0.3*A1*A2

    # 纳什均衡
    a_star = 1 / 1.3

    # 左图: 势函数等高线
    ax = axes[0]
    c = ax.contour(A1, A2, Phi, levels=20, cmap='viridis')
    ax.clabel(c, inline=True, fontsize=8)
    ax.plot(a_star, a_star, 'r*', markersize=20, label='纳什均衡')
    ax.set_xlabel('玩家1策略 $a_1$', fontsize=12)
    ax.set_ylabel('玩家2策略 $a_2$', fontsize=12)
    ax.set_title('势函数 $\\Phi(a_1, a_2)$\n（纳什均衡=势函数最小值）', fontsize=11)
    ax.legend()
    ax.set_aspect('equal')

    # 中图: 迭代最优响应轨迹
    ax = axes[1]
    ax.contour(A1, A2, Phi, levels=15, cmap='Greys', alpha=0.5)

    # 模拟迭代最优响应
    def br1(a2):
        return 1 - 0.3*a2
    def br2(a1):
        return 1 - 0.3*a1

    # 轨迹1: 从(0, 2)开始
    traj1 = [(0, 2)]
    for _ in range(8):
        a1_new = br1(traj1[-1][1])
        traj1.append((a1_new, traj1[-1][1]))
        a2_new = br2(a1_new)
        traj1.append((a1_new, a2_new))
    traj1 = np.array(traj1)
    ax.plot(traj1[:, 0], traj1[:, 1], 'b.-', linewidth=2, markersize=8, label='轨迹1')

    # 轨迹2: 从(2, 0)开始
    traj2 = [(2, 0)]
    for _ in range(8):
        a1_new = br1(traj2[-1][1])
        traj2.append((a1_new, traj2[-1][1]))
        a2_new = br2(a1_new)
        traj2.append((a1_new, a2_new))
    traj2 = np.array(traj2)
    ax.plot(traj2[:, 0], traj2[:, 1], 'g.-', linewidth=2, markersize=8, label='轨迹2')

    ax.plot(a_star, a_star, 'r*', markersize=20)
    ax.set_xlabel('玩家1策略 $a_1$', fontsize=12)
    ax.set_ylabel('玩家2策略 $a_2$', fontsize=12)
    ax.set_title('最优响应动力学\n（所有轨迹收敛到纳什均衡）', fontsize=11)
    ax.legend()
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # 右图: 势函数沿轨迹的变化
    ax = axes[2]
    phi_traj1 = [0.5*(a[0]-1)**2 + 0.5*(a[1]-1)**2 + 0.3*a[0]*a[1] for a in traj1]
    phi_traj2 = [0.5*(a[0]-1)**2 + 0.5*(a[1]-1)**2 + 0.3*a[0]*a[1] for a in traj2]

    ax.plot(phi_traj1, 'b.-', linewidth=2, markersize=8, label='轨迹1')
    ax.plot(phi_traj2, 'g.-', linewidth=2, markersize=8, label='轨迹2')
    ax.axhline(y=0.5*(a_star-1)**2 + 0.5*(a_star-1)**2 + 0.3*a_star*a_star,
              color='r', linestyle='--', label='最小值(纳什均衡)')

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('势函数 $\\Phi$', fontsize=12)
    ax.set_title('势函数单调下降\n（保证收敛!）', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/potential_game.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/potential_game.png', bbox_inches='tight')
    plt.close()
    print("Generated: potential_game.pdf/png")


# ==============================================================================
# 图7: CTDE范式对比
# ==============================================================================
def plot_ctde_comparison():
    """可视化CTDE与其他范式的对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 共同设置
    def draw_agents(ax, n=3, y_base=3, color='lightblue'):
        xs = np.linspace(1, 5, n)
        for i, x in enumerate(xs):
            circle = plt.Circle((x, y_base), 0.4, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y_base, f'$\\pi_{i+1}$', ha='center', va='center', fontsize=10)
        return xs

    # 左图: 完全独立学习 (IQL)
    ax = axes[0]
    xs = draw_agents(ax, color='lightyellow')

    # 环境
    rect = plt.Rectangle((0.5, 0.5), 5, 1, facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 1, '环境（非平稳!）', ha='center', va='center', fontsize=10)

    # 箭头: 独立交互
    for x in xs:
        ax.annotate('', xy=(x, 1.5), xytext=(x, 2.6),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

    ax.text(3, 4.5, '独立学习者', ha='center', fontsize=12, fontweight='bold')
    ax.text(3, -0.5, '问题: 环境对每个\n智能体是非平稳的',
           ha='center', fontsize=9, color='red')

    ax.set_xlim(0, 6)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('独立Q学习 (IQL)\n（分散训练、分散执行）', fontsize=11)

    # 中图: CTDE
    ax = axes[1]
    xs = draw_agents(ax, y_base=3.5, color='lightgreen')

    # 集中式Critic
    rect = plt.Rectangle((1, 1.5), 4, 1, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 2, '集中式Critic\n$V(s)$ 或 $Q_{tot}(s, \\mathbf{a})$',
           ha='center', va='center', fontsize=10)

    # 训练时的连接
    for x in xs:
        ax.annotate('', xy=(3, 2.5), xytext=(x, 3.1),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # 全局状态
    ax.text(5.5, 2, '$s$', fontsize=12, color='green')
    ax.annotate('', xy=(5, 2), xytext=(5.3, 2),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    ax.text(3, 4.8, '训练: 使用全局信息', ha='center', fontsize=10, color='blue')
    ax.text(3, 0.5, '执行: 仅用局部观测', ha='center', fontsize=10, color='green')

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CTDE范式\n（集中训练、分散执行）', fontsize=11)

    # 右图: 完全集中式
    ax = axes[2]

    # 单一策略
    rect = plt.Rectangle((2, 3), 2, 1, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 3.5, '中央\n控制器', ha='center', va='center', fontsize=10)

    # 智能体作为执行器
    xs = np.linspace(1, 5, 3)
    for i, x in enumerate(xs):
        circle = plt.Circle((x, 1.5), 0.3, facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(x, 1.5, f'{i+1}', ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(x, 1.8), xytext=(3, 3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.text(3, 4.8, '完全集中式', ha='center', fontsize=12, fontweight='bold')
    ax.text(3, 0.3, '问题: 可扩展性差,\n通信瓶颈',
           ha='center', fontsize=9, color='red')

    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('完全集中式\n（单一控制器）', fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/user/Lecture_15/figs_chap10/ctde_comparison.pdf', bbox_inches='tight')
    plt.savefig('/home/user/Lecture_15/figs_chap10/ctde_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated: ctde_comparison.pdf/png")


# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == "__main__":
    print("正在生成第10章图表...")
    print("=" * 50)

    plot_prisoners_dilemma()
    avg_with, avg_without = plot_braess_paradox()
    print(f"  Braess悖论: 有免费道路 = {avg_with:.0f}分钟, 无免费道路 = {avg_without:.0f}分钟")

    plot_network_game()
    plot_mfg_solution()
    plot_qmix_architecture()
    plot_potential_game()
    plot_ctde_comparison()

    print("=" * 50)
    print("所有图表生成完成!")
    print("位置: /home/user/Lecture_15/figs_chap10/")
