#!/usr/bin/env python3
"""
第9章：最短路与 5×5 GridWorld 上的值迭代 / Q-Learning / SARSA
================================================================

从仓库根目录运行::

    python3 code/chap09/chap09_rl.py

输出 6 张单独的图（每张同时有 pdf 与 png），与正文 \\includegraphics 一一对应：

    figures/chap09/chap09_shortest_path     正文最短路例子（s→a→c→t，代价 4）
    figures/chap09/chap09_value_iteration   几个状态的 V(s) 随值迭代轮数的变化
    figures/chap09/chap09_value_heatmap     收敛后的值函数热力图
    figures/chap09/chap09_policy            值迭代得到的最优策略（箭头）
    figures/chap09/chap09_qlearning_sarsa   Q-Learning 与 SARSA 的学习曲线（10 个随机种子）
    figures/chap09/chap09_qtable            Q-Learning 学到的 max_a Q(s,a) 热力图

环境设定与正文完全一致：5×5 网格，起点 (0,0)，终点 (4,4)，每一步奖励 −1，
到达终点回合结束，折扣 γ = 1（回合制任务）。于是最优值函数就是
"到终点曼哈顿距离的负值"，起点 V = −8。
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, COLORS  # noqa: E402

setup_style()

# 值函数热力图共用的色带：越绿表示回报越高（离终点越近）
GREEN_CMAP = LinearSegmentedColormap.from_list(
    'value_green', ['#ffffff', '#cfe6d1', '#8fc497', COLORS['green'], '#1f4d26'])
V_MIN, V_MAX = -8, 0

# ============================================================================
# Part 1：最短路问题（正文 TikZ 图中的有向图）
# ============================================================================
print("=" * 60)
print("Part 1：最短路问题（正文例子）")
print("=" * 60)

graph = {
    's': [('a', 2), ('b', 5)],
    'a': [('c', 1), ('b', 3)],
    'b': [('c', 2), ('t', 6)],
    'c': [('t', 1)],
    't': [],
}
nodes = ['s', 'a', 'b', 'c', 't']


def shortest_path_dp(graph, end='t'):
    """贝尔曼方程 d(v) = min_u [w_vu + d(u)] 的逐轮更新（Bellman-Ford 形式）。"""
    V = {n: np.inf for n in nodes}
    V[end] = 0.0
    policy = {n: None for n in nodes}
    for it in range(len(nodes)):
        V_old = dict(V)
        for v in nodes:
            if v == end:
                continue
            best = min(((w + V_old[u], u) for u, w in graph[v]), key=lambda p: p[0])
            V[v], policy[v] = best
        if V == V_old:
            print(f"值更新在第 {it + 1} 轮收敛")
            break
    return V, policy


V_sp, policy_sp = shortest_path_dp(graph)
for n in nodes:
    print(f"  V({n}) = {V_sp[n]:.0f}" + (f"，下一步 -> {policy_sp[n]}" if policy_sp[n] else ""))
path = ['s']
while path[-1] != 't':
    path.append(policy_sp[path[-1]])
print("最短路径：", " -> ".join(path), f"，代价 = {V_sp['s']:.0f}")


def plot_shortest_path():
    """有向图 + 边权 + 每个节点的 V 值；最优路径红色。"""
    pos = {'s': (0.0, 0.0), 'a': (2.5, 1.5), 'b': (2.5, -1.5), 'c': (5.0, 0.0), 't': (7.5, 0.0)}
    R = 0.40  # 节点半径
    edges = [('s', 'a', 2), ('s', 'b', 5), ('a', 'c', 1), ('a', 'b', 3),
             ('b', 'c', 2), ('c', 't', 1), ('b', 't', 6)]
    path_edges = {('s', 'a'), ('a', 'c'), ('c', 't')}
    # 边权标签相对边中点的偏移（沿用 TikZ 里 above/below/left/right 的放置）
    label_offset = {('s', 'a'): (-0.22, 0.22), ('s', 'b'): (-0.22, -0.22), ('a', 'c'): (0.0, 0.28),
                    ('a', 'b'): (0.28, 0.0), ('b', 'c'): (0.12, -0.28), ('c', 't'): (0.0, 0.28),
                    ('b', 't'): (0.0, -0.42)}
    # V 值标签的位置
    v_label = {'s': (-0.55, 0.0, 'right', 'center'), 'a': (2.5, 2.05, 'center', 'bottom'),
               'b': (2.5, -2.05, 'center', 'top'), 'c': (5.0, 0.55, 'center', 'bottom'),
               't': (8.05, 0.0, 'left', 'center')}
    fill = {'s': '#dbe7f3', 't': '#d7ecd9'}

    fig, ax = plt.subplots(figsize=(3.9, 2.55))
    for u, v, w in edges:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        p_start = (x1 + R * ux, y1 + R * uy)
        p_end = (x2 - R * ux, y2 - R * uy)
        on_path = (u, v) in path_edges
        color = COLORS['red'] if on_path else COLORS['black']
        lw = 2.2 if on_path else 1.1
        rad = 0.28 if (u, v) == ('b', 't') else 0.0
        arrow = FancyArrowPatch(p_start, p_end, arrowstyle='-|>', mutation_scale=9,
                                color=color, lw=lw, zorder=2,
                                connectionstyle=f'arc3,rad={rad}')
        ax.add_patch(arrow)
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if rad:  # arc3 二次 Bézier 曲线的中点 = 弦中点 + 0.5·rad·(dy, −dx)
            mx, my = mx + 0.5 * rad * dy, my - 0.5 * rad * dx
        ox, oy = label_offset[(u, v)]
        ax.text(mx + ox, my + oy, str(w), fontsize=9, ha='center', va='center',
                color=color, fontweight='bold' if on_path else 'normal',
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), R, fc=fill.get(n, 'white'), ec=COLORS['black'], lw=1.1, zorder=3))
        ax.text(x, y, f'${n}$', fontsize=10.5, ha='center', va='center', zorder=4)
        lx, ly, ha, va = v_label[n]
        ax.text(lx, ly, f'$V={V_sp[n]:.0f}$', fontsize=8.5, ha=ha, va=va,
                color=COLORS['blue'], zorder=4)
    ax.text(0.0, -0.62, '起点', fontsize=8, ha='center', va='top', color=COLORS['gray'])
    ax.text(7.5, -0.62, '终点', fontsize=8, ha='center', va='top', color=COLORS['gray'])
    ax.set_xlim(-1.3, 8.9)
    ax.set_ylim(-2.55, 2.55)
    ax.set_aspect('equal')
    ax.axis('off')
    save_figure(fig, 'figures/chap09/chap09_shortest_path')


plot_shortest_path()


# ============================================================================
# Part 2：5×5 GridWorld 与值迭代（有模型）
# ============================================================================
print("\n" + "=" * 60)
print("Part 2：5×5 GridWorld 上的值迭代")
print("=" * 60)


class GridWorld:
    """与正文代码相同的 5×5 网格世界：每步奖励 −1，到达 (4,4) 结束。"""

    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

    def step(self, state, action):
        i, j = state
        di, dj = self.actions[action]
        ni = max(0, min(i + di, self.size - 1))
        nj = max(0, min(j + dj, self.size - 1))
        new_state = (ni, nj)
        reward = -1
        done = (new_state == self.goal)
        return new_state, reward, done


def value_iteration(env, gamma=1.0, threshold=1e-6, max_iter=1000):
    """返回收敛的 V、每一轮之后的 V 快照（含初始的全 0）、收敛轮数。"""
    V = np.zeros((env.size, env.size))
    history = [V.copy()]
    for it in range(max_iter):
        delta = 0.0
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) == env.goal:
                    continue
                q_values = []
                for a in range(4):
                    s2, r, done = env.step((i, j), a)
                    q_values.append(r + (0.0 if done else gamma * V[s2]))
                new_v = max(q_values)
                delta = max(delta, abs(new_v - V[i, j]))
                V[i, j] = new_v
        history.append(V.copy())
        if delta < threshold:
            print(f"值迭代：{it + 1} 轮后收敛")
            break
    return V, history


def greedy_actions(env, V, gamma=1.0, tol=1e-9):
    """每个状态所有并列最优的动作（曼哈顿网格上经常"下"和"右"一样好）。"""
    best = {}
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.goal:
                continue
            q = []
            for a in range(4):
                s2, r, done = env.step((i, j), a)
                q.append(r + (0.0 if done else gamma * V[s2]))
            q = np.array(q)
            best[(i, j)] = [a for a in range(4) if q[a] >= q.max() - tol]
    return best


GAMMA = 1.0
env = GridWorld(size=5)
V, V_history = value_iteration(env, gamma=GAMMA)
policy_sets = greedy_actions(env, V, gamma=GAMMA)
print("值函数 V(s)：\n", V.round(2))

# ---- 图：若干状态的 V 随迭代轮数变化 ----
fig, ax = plt.subplots(figsize=(3.1, 2.45))
watch = [(0, 0), (1, 1), (2, 2), (3, 3), (3, 4)]
markers = ['o', 's', '^', 'D', 'v']
for (i, j), mk in zip(watch, markers):
    vals = [h[i, j] for h in V_history]
    ax.plot(range(len(vals)), vals, marker=mk, markersize=3.5, lw=1.4,
            label=f'$s=({i},{j})$')
ax.set_xlabel('值迭代轮数 $k$')
ax.set_ylabel('$V_k(s)$')
ax.set_xticks(range(len(V_history)))
ax.set_yticks(range(-8, 1, 2))
ax.legend(fontsize=7.5, loc='lower left', ncol=1, handlelength=1.6)
save_figure(fig, 'figures/chap09/chap09_value_iteration')


def value_heatmap(values, path_no_ext, cbar_label):
    """5×5 值热力图：格子里标数值，终点标"目标"。"""
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    im = ax.imshow(values, cmap=GREEN_CMAP, vmin=V_MIN, vmax=V_MAX, interpolation='nearest')
    for i in range(env.size):
        for j in range(env.size):
            txt_color = 'white' if values[i, j] > -2.5 else COLORS['black']
            if (i, j) == env.goal:
                ax.text(j, i, '目标\n$0$', ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold', linespacing=1.1)
            else:
                ax.text(j, i, f'${values[i, j]:.1f}$', ha='center', va='center',
                        fontsize=8, color=txt_color)
    ax.text(-0.42, -0.42, '起点', ha='left', va='top', fontsize=6.5, color=COLORS['gray'])
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlabel('列 $j$', labelpad=2)
    ax.set_ylabel('行 $i$', labelpad=2)
    ax.grid(False)
    ax.tick_params(length=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    save_figure(fig, path_no_ext)


value_heatmap(V, 'figures/chap09/chap09_value_heatmap', '$V(s)$')

# ---- 图：最优策略箭头 ----
fig, ax = plt.subplots(figsize=(2.85, 2.85))
arrow_dir = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}  # 画图坐标：y 向上为"上"
for i in range(env.size):
    for j in range(env.size):
        x, y = j, -i
        if (i, j) == env.goal:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc='#d7ecd9', ec=COLORS['gray'], lw=0.8))
            ax.text(x, y, '目标', ha='center', va='center', fontsize=8.5, fontweight='bold',
                    color=COLORS['green'])
            continue
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc='white', ec=COLORS['gray'], lw=0.8))
        for a in policy_sets[(i, j)]:
            dx, dy = arrow_dir[a]
            ax.add_patch(FancyArrowPatch((x - 0.02 * dx, y - 0.02 * dy), (x + 0.36 * dx, y + 0.36 * dy),
                                         arrowstyle='-|>', mutation_scale=8, color=COLORS['red'],
                                         lw=1.5, zorder=3))
        ax.plot(x, y, 'o', color=COLORS['red'], markersize=2.2, zorder=4)
ax.text(-0.45, 0.45, '起点', ha='left', va='top', fontsize=6.5, color=COLORS['gray'])
ax.set_xlim(-0.5, env.size - 0.5)
ax.set_ylim(-(env.size - 0.5), 0.5)
ax.set_xticks(range(env.size))
ax.set_yticks([-i for i in range(env.size)])
ax.set_yticklabels(range(env.size))
ax.set_xlabel('列 $j$', labelpad=2)
ax.set_ylabel('行 $i$', labelpad=2)
ax.set_aspect('equal')
ax.grid(False)
ax.tick_params(length=2)
for sp in ax.spines.values():
    sp.set_visible(False)
save_figure(fig, 'figures/chap09/chap09_policy')


# ============================================================================
# Part 3：Q-Learning 与 SARSA（无模型）
# ============================================================================
print("\n" + "=" * 60)
print("Part 3：Q-Learning 与 SARSA（无模型，10 个随机种子）")
print("=" * 60)


def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q >= q.max() - 1e-12)))


def q_learning(env, episodes=2000, gamma=1.0, alpha=0.1, epsilon=0.1, seed=0, max_steps=500):
    """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]（off-policy）。"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.size, env.size, 4))
    returns = np.zeros(episodes)
    for ep in range(episodes):
        s, total = env.start, 0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, done = env.step(s, a)
            target = r if done else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            total += r
            s = s2
            if done:
                break
        returns[ep] = total
    return Q, returns


def sarsa(env, episodes=2000, gamma=1.0, alpha=0.1, epsilon=0.1, seed=0, max_steps=500):
    """Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]，a' 是真的会执行的动作（on-policy）。"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.size, env.size, 4))
    returns = np.zeros(episodes)
    for ep in range(episodes):
        s, total = env.start, 0
        a = epsilon_greedy(Q, s, epsilon, rng)
        for _ in range(max_steps):
            s2, r, done = env.step(s, a)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            target = r if done else r + gamma * Q[s2][a2]
            Q[s][a] += alpha * (target - Q[s][a])
            total += r
            s, a = s2, a2
            if done:
                break
        returns[ep] = total
    return Q, returns


EPISODES, SEEDS, WINDOW = 2000, 10, 50
ql_runs, sa_runs, Q_tables = [], [], []
for seed in range(SEEDS):
    Q_ql, r_ql = q_learning(env, EPISODES, GAMMA, seed=seed)
    Q_sa, r_sa = sarsa(env, EPISODES, GAMMA, seed=seed)
    ql_runs.append(r_ql)
    sa_runs.append(r_sa)
    Q_tables.append(Q_ql)
ql_runs, sa_runs = np.array(ql_runs), np.array(sa_runs)
print(f"Q-Learning 最后 100 回合平均回报：{ql_runs[:, -100:].mean():.2f} ± {ql_runs[:, -100:].mean(1).std():.2f}")
print(f"SARSA      最后 100 回合平均回报：{sa_runs[:, -100:].mean():.2f} ± {sa_runs[:, -100:].mean(1).std():.2f}")
print(f"前 50 回合平均回报：Q-Learning {ql_runs[:, :50].mean():.1f}，SARSA {sa_runs[:, :50].mean():.1f}")

Q_learned = Q_tables[0]
Q_max = Q_learned.max(axis=2)
print("Q-Learning（种子 0）学到的 max_a Q(s,a)：\n", Q_max.round(2))
print(f"与值迭代 V 的最大偏差：{np.abs(Q_max - V).max():.3f}")


def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# ---- 图：学习曲线 ----
fig, ax = plt.subplots(figsize=(3.5, 2.5))
x_ma = np.arange(WINDOW, EPISODES + 1)
for runs, color, name in [(ql_runs, COLORS['blue'], 'Q-Learning'),
                          (sa_runs, COLORS['orange'], 'SARSA')]:
    ma = np.array([moving_average(r, WINDOW) for r in runs])
    m, sd = ma.mean(0), ma.std(0)
    ax.plot(x_ma, m, color=color, lw=1.5, label=name)
    ax.fill_between(x_ma, m - sd, m + sd, color=color, alpha=0.2, lw=0)
ax.axhline(-8, color=COLORS['black'], ls='--', lw=1.0, label='最优回报 $-8$')
ax.set_yscale('symlog', linthresh=10)
ax.set_ylim(ql_runs.min(), 0)
ax.set_yticks([0, -5, -10, -20, -50, -100, -200])
ax.set_yticklabels(['0', '$-5$', '$-10$', '$-20$', '$-50$', '$-100$', '$-200$'])
ax.set_xlim(0, EPISODES)
ax.set_xlabel('回合数')
ax.set_ylabel(f'回合总奖励（{WINDOW} 回合滑动平均）')
ax.legend(loc='lower right', fontsize=8)
save_figure(fig, 'figures/chap09/chap09_qlearning_sarsa')

# ---- 图：Q 表热力图 ----
value_heatmap(Q_max, 'figures/chap09/chap09_qtable', r'$\max_a Q(s,a)$')

print("\n第9章代码执行完成！")
