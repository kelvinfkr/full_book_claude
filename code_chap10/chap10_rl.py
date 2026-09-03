#!/usr/bin/env python3
"""
第10章：强化学习与最短路问题演示

演示：
1. 最短路问题的动态规划求解
2. 值迭代算法
3. 策略迭代算法
"""
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第10章：强化学习与贝尔曼方程演示")
print("="*60)

# ==================== 最短路问题 ====================
print("\n" + "="*60)
print("Part 1: 最短路问题（书中例子）")
print("="*60)

# 图结构（与书中一致）
graph = {
    's': [('a', 2), ('b', 5)],
    'a': [('c', 1), ('b', 3)],
    'b': [('c', 2), ('t', 6)],
    'c': [('t', 1)],
    't': []
}

nodes = ['s', 'a', 'b', 'c', 't']

def shortest_path_dp(graph, start, end):
    """动态规划求最短路"""
    INF = float('inf')
    V = {node: INF for node in nodes}
    V[end] = 0
    policy = {node: None for node in nodes}

    iterations = []
    for iteration in range(10):
        V_old = V.copy()
        for node in nodes:
            if node == end:
                continue
            min_val = INF
            best_next = None
            for (next_node, cost) in graph[node]:
                val = cost + V_old[next_node]
                if val < min_val:
                    min_val = val
                    best_next = next_node
            V[node] = min_val
            policy[node] = best_next
        iterations.append(V.copy())
        if V == V_old:
            print(f"值迭代在第 {iteration+1} 轮收敛")
            break
    return V, policy, iterations

V_sp, policy_sp, iterations = shortest_path_dp(graph, 's', 't')

print("\n最短距离 V(node):")
for node in nodes:
    print(f"  V({node}) = {V_sp[node]}")

print("\n最优策略:")
for node in nodes:
    if policy_sp[node]:
        print(f"  从 {node} -> {policy_sp[node]}")

path = ['s']
current = 's'
while current != 't':
    current = policy_sp[current]
    path.append(current)
print(f"\n最短路径: {' -> '.join(path)}, 代价={V_sp['s']}")

# ==================== 网格世界MDP ====================
print("\n" + "="*60)
print("Part 2: 网格世界MDP")
print("="*60)

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = size * size - 1

    def step(self, state, action):
        row, col = state // self.size, state % self.size
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        next_state = row * self.size + col
        reward = 0 if next_state == self.goal else -1
        return next_state, reward

    def is_terminal(self, state):
        return state == self.goal

def value_iteration(env, gamma=0.99, theta=1e-6, max_iter=1000):
    V = np.zeros(env.n_states)
    V_history = [V.copy()]

    for i in range(max_iter):
        delta = 0
        V_new = V.copy()
        for s in range(env.n_states):
            if env.is_terminal(s): continue
            values = []
            for a in range(env.n_actions):
                s_next, r = env.step(s, a)
                values.append(r + gamma * V[s_next])
            V_new[s] = max(values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        V_history.append(V.copy())
        if delta < theta:
            print(f"值迭代在 {i+1} 次迭代后收敛")
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if env.is_terminal(s): continue
        values = []
        for a in range(env.n_actions):
            s_next, r = env.step(s, a)
            values.append(r + gamma * V[s_next])
        policy[s] = np.argmax(values)
    return V, policy, V_history

env = GridWorld(size=4)
V, policy, V_history = value_iteration(env, gamma=0.99)

print("\n值函数 (4x4网格):")
print(V.reshape(4, 4).round(2))

# ==================== Q-Learning ====================
print("\n" + "="*60)
print("Part 3: Q-Learning (无模型强化学习)")
print("="*60)

print("""
Q-Learning 核心思想：
- 不需要知道环境的转移概率（无模型）
- 通过与环境交互，不断更新 Q(s,a) 表
- 使用 Bellman 方程的采样版本：
  Q(s,a) ← Q(s,a) + α * [r + γ max Q(s',a') - Q(s,a)]
""")

def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Q-Learning 算法

    参数说明：
    - env: 环境（必须有 step, reset, is_terminal 方法）
    - episodes: 训练回合数
    - alpha: 学习率，控制新信息覆盖旧信息的程度
            α太大→学习不稳定，α太小→学习太慢
    - gamma: 折扣因子，γ接近1表示更看重长期回报
    - epsilon: 探索率，ε-贪婪策略中随机探索的概率

    返回：Q表、每回合总奖励列表
    """
    # 初始化 Q 表（状态 × 动作）
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for episode in range(episodes):
        # 每个回合从状态0开始
        state = 0
        total_reward = 0
        steps = 0
        max_steps = 100  # 防止无限循环

        while not env.is_terminal(state) and steps < max_steps:
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)  # 探索
            else:
                action = np.argmax(Q[state])  # 利用

            # 执行动作，观察结果
            next_state, reward = env.step(state, action)
            total_reward += reward

            # Q-Learning 更新公式（核心！）
            # 这是 Bellman 方程的随机近似
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            steps += 1

        episode_rewards.append(total_reward)

        # 每100回合打印一次
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}: 平均奖励 = {avg_reward:.2f}")

    return Q, episode_rewards

# 训练 Q-Learning
print("\n开始 Q-Learning 训练...")
Q_learned, rewards = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)

print("\n学到的 Q 表 (部分):")
print("状态 | 上    下    左    右")
print("-" * 35)
for s in [0, 1, 4, 5, 10, 14]:
    print(f"  {s:2d} | {Q_learned[s, 0]:5.2f} {Q_learned[s, 1]:5.2f} {Q_learned[s, 2]:5.2f} {Q_learned[s, 3]:5.2f}")

# 从 Q 表提取最优策略
q_policy = np.argmax(Q_learned, axis=1)
action_names = ['↑', '↓', '←', '→']
print("\nQ-Learning 学到的策略:")
for i in range(4):
    row = ""
    for j in range(4):
        s = i * 4 + j
        if env.is_terminal(s):
            row += " G "
        else:
            row += f" {action_names[q_policy[s]]} "
    print(row)

# ==================== SARSA对比 ====================
print("\n" + "="*60)
print("Part 4: SARSA vs Q-Learning 对比")
print("="*60)

print("""
SARSA 和 Q-Learning 的区别：
- Q-Learning (off-policy):
  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
  更新时用的是 s' 的最大 Q 值（不管实际采取什么动作）

- SARSA (on-policy):
  Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
  更新时用的是实际在 s' 采取的动作 a' 的 Q 值

简单说：Q-Learning 更"乐观"（假设未来会选最优），SARSA 更"现实"（用实际策略）
""")

def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA 算法"""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for episode in range(episodes):
        state = 0
        # SARSA 需要先选好动作
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        total_reward = 0
        steps = 0

        while not env.is_terminal(state) and steps < 100:
            next_state, reward = env.step(state, action)
            total_reward += reward

            # SARSA: 先选好下一个动作
            if np.random.random() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA 更新：用实际的 next_action
            td_target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action
            steps += 1

        episode_rewards.append(total_reward)

    return Q, episode_rewards

print("训练 SARSA...")
Q_sarsa, rewards_sarsa = sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
print(f"SARSA 最后100回合平均奖励: {np.mean(rewards_sarsa[-100:]):.2f}")
print(f"Q-Learning 最后100回合平均奖励: {np.mean(rewards[-100:]):.2f}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 图1：最短路图
ax1 = axes[0, 0]
ax1.set_xlim(-0.5, 4)
ax1.set_ylim(-0.5, 3)

pos = {'s': (0, 1.5), 'a': (1.5, 2.5), 'b': (1.5, 0.5), 'c': (2.5, 1.5), 't': (3.5, 1.5)}

for node, (x, y) in pos.items():
    color = 'lightblue' if node not in ['s', 't'] else ('lightgreen' if node == 't' else 'lightyellow')
    circle = plt.Circle((x, y), 0.25, color=color, ec='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, y, node, ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(x, y-0.4, f'V={V_sp[node]}', ha='center', va='center', fontsize=10, color='red')

edges = [('s', 'a', 2), ('s', 'b', 5), ('a', 'c', 1), ('a', 'b', 3), ('b', 'c', 2), ('b', 't', 6), ('c', 't', 1)]
for start, end, cost in edges:
    x1, y1 = pos[start]
    x2, y2 = pos[end]
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length, dy/length
    ax1.annotate('', xy=(x2-0.3*dx, y2-0.3*dy), xytext=(x1+0.3*dx, y1+0.3*dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax1.text(mx+0.1, my+0.1, str(cost), fontsize=11, color='blue')

path_edges = [('s', 'a'), ('a', 'c'), ('c', 't')]
for start, end in path_edges:
    x1, y1 = pos[start]
    x2, y2 = pos[end]
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length, dy/length
    ax1.annotate('', xy=(x2-0.3*dx, y2-0.3*dy), xytext=(x1+0.3*dx, y1+0.3*dy),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))

ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('最短路问题 (书中例子)\n最优: s -> a -> c -> t (代价=4)', fontsize=14)

# 图2：值迭代收敛
ax2 = axes[0, 1]
for s in [0, 5, 10]:
    values = [V_history[i][s] for i in range(len(V_history))]
    ax2.plot(values, label=f'状态 {s}', linewidth=2)
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('值 V(s)', fontsize=12, labelpad=10)
ax2.set_title('值迭代收敛过程\n(4x4网格世界)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：值函数热力图
ax3 = axes[1, 0]
im = ax3.imshow(V.reshape(4, 4), cmap='RdYlGn')
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))
for i in range(4):
    for j in range(4):
        ax3.text(j, i, f'{V[i*4+j]:.1f}', ha='center', va='center', fontsize=11)
ax3.set_title('值函数 V(s)\n(4x4网格, 目标在右下角)', fontsize=14)
plt.colorbar(im, ax=ax3)

# 图4：最优策略
ax4 = axes[1, 1]
arrow_dx = {0: 0, 1: 0, 2: -0.3, 3: 0.3}
arrow_dy = {0: 0.3, 1: -0.3, 2: 0, 3: 0}

ax4.set_xlim(-0.5, 3.5)
ax4.set_ylim(-0.5, 3.5)

for i in range(4):
    for j in range(4):
        s = i * 4 + j
        if env.is_terminal(s):
            ax4.add_patch(plt.Circle((j, 3-i), 0.4, color='lightgreen', ec='black', linewidth=2))
            ax4.text(j, 3-i, '目标', ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax4.add_patch(plt.Rectangle((j-0.4, 3-i-0.4), 0.8, 0.8,
                                        color='lightyellow', ec='black', linewidth=1))
            a = policy[s]
            ax4.annotate('', xy=(j+arrow_dx[a], 3-i+arrow_dy[a]),
                        xytext=(j, 3-i),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax4.set_aspect('equal')
ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
ax4.set_yticklabels([3, 2, 1, 0])
ax4.set_title('值迭代最优策略\n(箭头表示各状态的最佳动作)', fontsize=14)
ax4.grid(True, alpha=0.3)

# 图5：Q-Learning 学习曲线
ax5 = axes[0, 2]
window = 20
smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax5.plot(smoothed_rewards, 'b-', linewidth=1.5, label='Q-Learning')
smoothed_sarsa = np.convolve(rewards_sarsa, np.ones(window)/window, mode='valid')
ax5.plot(smoothed_sarsa, 'r-', linewidth=1.5, alpha=0.7, label='SARSA')
ax5.set_xlabel('回合数', fontsize=12)
ax5.set_ylabel('平均奖励 (20回合滑动窗口)', fontsize=12, labelpad=10)
ax5.set_title('Q-Learning vs SARSA\n学习曲线对比', fontsize=14)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 图6：Q 表可视化（热力图）
ax6 = axes[1, 2]
Q_max = np.max(Q_learned, axis=1).reshape(4, 4)
im6 = ax6.imshow(Q_max, cmap='Blues')
ax6.set_xticks(range(4))
ax6.set_yticks(range(4))
for i in range(4):
    for j in range(4):
        ax6.text(j, i, f'{Q_max[i, j]:.1f}', ha='center', va='center', fontsize=10)
ax6.set_title('Q-Learning 学到的 max Q(s,a)\n(通过与环境交互学习)', fontsize=14)
plt.colorbar(im6, ax=ax6)

plt.tight_layout()
plt.savefig('figs/chap10_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap10_fig1.png")

plt.show()
print("\n第10章代码执行完成！")
