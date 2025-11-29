#!/usr/bin/env python3
"""
生成chap16的所有可视化图表
包括：世界模型、CBF安全控制器、可微物理仿真
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

import os

# 使用相对路径
output_dir = 'figs_chap16'
os.makedirs(output_dir, exist_ok=True)

def save_fig(name):
    """同时保存PDF和PNG"""
    plt.savefig(f'{output_dir}/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {name}.pdf, {name}.png")

# =============================================================================
# 图1: 生成式vs JEPA范式对比
# =============================================================================
def fig_paradigm_comparison():
    """生成两种范式对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 100)
    x_true = t
    y_true = 5 - 0.5 * (t - np.pi)**2
    y_pred = y_true + 0.3 * np.random.randn(len(t))

    ax1.plot(x_true, y_true, 'b-', linewidth=3, label='真实轨迹')
    ax1.scatter(x_true[::5], y_pred[::5], c='r', s=50, alpha=0.6, label='像素预测')
    ax1.fill_between(x_true, y_pred - 0.5, y_pred + 0.5, alpha=0.2, color='red')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('位置', fontsize=12)
    ax1.set_title('生成式模型：预测像素\n（高维、有噪声）', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    z1 = np.cos(t) + 0.05 * np.random.randn(len(t))
    z2 = np.sin(t) + 0.05 * np.random.randn(len(t))
    z1_pred = np.cos(t + 0.1)
    z2_pred = np.sin(t + 0.1)

    ax2.plot(z1, z2, 'b-', linewidth=3, label='真实潜空间轨迹')
    ax2.plot(z1_pred, z2_pred, 'r--', linewidth=2, label='预测轨迹')
    ax2.scatter([z1[0]], [z2[0]], c='green', s=100, zorder=5, marker='o', label='起点')
    ax2.scatter([z1[-1]], [z2[-1]], c='purple', s=100, zorder=5, marker='s', label='终点')
    ax2.set_xlabel('潜空间维度1', fontsize=12)
    ax2.set_ylabel('潜空间维度2', fontsize=12)
    ax2.set_title('JEPA：在潜空间预测\n（低维、本质）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    save_fig('paradigm_comparison')

# =============================================================================
# 图2: 潜空间动力学模型示意图
# =============================================================================
def fig_latent_dynamics():
    """展示潜空间动力学模型如何在"梦境"中规划"""
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 4)

    np.random.seed(123)
    T = 50
    t = np.arange(T)

    obs_dim = 10
    true_state = 5 * (1 - np.exp(-t / 15))
    observations = np.outer(true_state, np.ones(obs_dim)) + 0.5 * np.random.randn(T, obs_dim)

    im = ax1.imshow(observations.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('时间步', fontsize=11)
    ax1.set_ylabel('观测维度', fontsize=11)
    ax1.set_title('(a) 高维观测 (10维)', fontsize=12)
    plt.colorbar(im, ax=ax1)

    z1 = true_state / 5
    z2 = 0.3 * np.sin(2 * np.pi * t / T) + 0.1 * np.random.randn(T)

    ax2.scatter(z1, z2, c=t, cmap='plasma', s=30)
    ax2.plot(z1, z2, 'k-', alpha=0.3)
    ax2.scatter([z1[0]], [z2[0]], c='green', s=150, marker='o', label='起点', zorder=10)
    ax2.scatter([z1[-1]], [z2[-1]], c='red', s=150, marker='*', label='目标', zorder=10)
    ax2.set_xlabel('潜空间维度1（位置信息）', fontsize=11)
    ax2.set_ylabel('潜空间维度2（速度信息）', fontsize=11)
    ax2.set_title('(b) 潜空间 (2维) - 捕捉本质', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title('(c) "做梦"：想象的多条轨迹', fontsize=12)
    for i in range(5):
        noise = 0.3 * np.random.randn(T)
        imagined = true_state + noise.cumsum() * 0.1
        imagined = np.clip(imagined, 0, 7)
        alpha = 0.3 + 0.15 * i
        ax3.plot(t, imagined, alpha=alpha, linewidth=2, label=f'轨迹 {i+1}')
    ax3.axhline(y=5, color='r', linestyle='--', label='目标')
    ax3.set_xlabel('时间步', fontsize=11)
    ax3.set_ylabel('想象的位置', fontsize=11)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    ax4.set_title('(d) 各轨迹的预测奖励', fontsize=12)
    for i in range(5):
        rewards = -np.abs(true_state - 5 + 0.5 * np.random.randn(T))
        rewards = np.cumsum(rewards)
        ax4.plot(t, rewards, linewidth=2, label=f'轨迹 {i+1}')
    ax4.set_xlabel('时间步', fontsize=11)
    ax4.set_ylabel('累积奖励', fontsize=11)
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('latent_dynamics')

# =============================================================================
# 图3: CBF安全控制器演示
# =============================================================================
def fig_cbf_controller():
    """演示CBF控制器如何保证安全"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    obstacle_pos = np.array([5.0, 5.0])
    safe_distance = 1.5

    class CBFController:
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def barrier_function(self, x, obstacle_pos, safe_distance):
            return np.sum((x - obstacle_pos)**2) - safe_distance**2

        def barrier_gradient(self, x, obstacle_pos):
            return 2 * (x - obstacle_pos)

        def safe_control(self, x, u_ref, obstacle_pos, safe_distance):
            h = self.barrier_function(x, obstacle_pos, safe_distance)
            grad_h = self.barrier_gradient(x, obstacle_pos)
            if grad_h @ u_ref + self.alpha * h >= 0:
                return u_ref, False
            grad_h_norm_sq = np.sum(grad_h**2)
            if grad_h_norm_sq < 1e-10:
                return u_ref, False
            lam = -(grad_h @ u_ref + self.alpha * h) / grad_h_norm_sq
            lam = max(0, lam)
            u_safe = u_ref - lam * grad_h
            return u_safe, True

    controller = CBFController(alpha=1.0)
    dt = 0.1
    T = 100
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])

    x_unsafe = [start.copy()]
    x = start.copy()
    for _ in range(T):
        u_ref = 0.3 * (goal - x) / (np.linalg.norm(goal - x) + 0.1)
        x = x + dt * u_ref
        x_unsafe.append(x.copy())
    x_unsafe = np.array(x_unsafe)

    x_safe = [start.copy()]
    corrections = []
    x = start.copy()
    for _ in range(T):
        u_ref = 0.3 * (goal - x) / (np.linalg.norm(goal - x) + 0.1)
        u_safe, corrected = controller.safe_control(x, u_ref, obstacle_pos, safe_distance)
        x = x + dt * u_safe
        x_safe.append(x.copy())
        corrections.append(corrected)
    x_safe = np.array(x_safe)

    ax1 = axes[0, 0]
    circle = plt.Circle(obstacle_pos, safe_distance, color='red', alpha=0.3)
    ax1.add_patch(circle)
    ax1.plot(x_unsafe[:, 0], x_unsafe[:, 1], 'b-', linewidth=2, label='轨迹（无CBF）')
    ax1.scatter([start[0]], [start[1]], c='green', s=100, zorder=5, label='起点')
    ax1.scatter([goal[0]], [goal[1]], c='purple', s=100, marker='*', zorder=5, label='目标')
    ax1.scatter([obstacle_pos[0]], [obstacle_pos[1]], c='red', s=150, marker='x', zorder=5, label='障碍物')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('(a) 无CBF：发生碰撞！', fontsize=14)
    ax1.legend()
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    circle = plt.Circle(obstacle_pos, safe_distance, color='red', alpha=0.3)
    ax2.add_patch(circle)
    ax2.plot(x_safe[:, 0], x_safe[:, 1], 'g-', linewidth=2, label='轨迹（有CBF）')
    ax2.scatter([start[0]], [start[1]], c='green', s=100, zorder=5, label='起点')
    ax2.scatter([goal[0]], [goal[1]], c='purple', s=100, marker='*', zorder=5, label='目标')
    ax2.scatter([obstacle_pos[0]], [obstacle_pos[1]], c='red', s=150, marker='x', zorder=5, label='障碍物')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('(b) 有CBF：安全绕行！', fontsize=14)
    ax2.legend()
    ax2.set_xlim(-1, 12)
    ax2.set_ylim(-1, 12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    h_unsafe = [np.sum((x - obstacle_pos)**2) - safe_distance**2 for x in x_unsafe]
    h_safe = [np.sum((x - obstacle_pos)**2) - safe_distance**2 for x in x_safe]
    t = np.arange(len(h_unsafe))
    ax3.plot(t, h_unsafe, 'b-', linewidth=2, label='h(x) 无CBF')
    ax3.plot(t, h_safe, 'g-', linewidth=2, label='h(x) 有CBF')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='安全边界 (h=0)')
    ax3.fill_between(t, -10, 0, alpha=0.2, color='red', label='危险区域 (h<0)')
    ax3.set_xlabel('时间步', fontsize=12)
    ax3.set_ylabel('h(x) = ||x - obs||² - d²', fontsize=12)
    ax3.set_title('(c) 障碍函数值随时间变化', fontsize=14)
    ax3.legend()
    ax3.set_ylim(-10, 100)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    u_ref_list = []
    u_safe_list = []
    x = start.copy()
    for i in range(T):
        u_ref = 0.3 * (goal - x) / (np.linalg.norm(goal - x) + 0.1)
        u_safe, _ = controller.safe_control(x, u_ref, obstacle_pos, safe_distance)
        u_ref_list.append(np.linalg.norm(u_ref))
        u_safe_list.append(np.linalg.norm(u_safe))
        x = x + dt * u_safe

    t = np.arange(T)
    ax4.plot(t, u_ref_list, 'b-', linewidth=2, label='||u_ref|| (期望)')
    ax4.plot(t, u_safe_list, 'g-', linewidth=2, label='||u_safe|| (实际)')
    ax4.fill_between(t, u_safe_list, u_ref_list, where=np.array(corrections),
                     alpha=0.3, color='orange', label='CBF修正激活')
    ax4.set_xlabel('时间步', fontsize=12)
    ax4.set_ylabel('控制量大小', fontsize=12)
    ax4.set_title('(d) 控制输入：CBF仅在必要时修正', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('cbf_controller')

# =============================================================================
# 图4: 多障碍物避障
# =============================================================================
def fig_multi_obstacle():
    """演示多障碍物环境下的CBF控制"""
    fig, ax = plt.subplots(figsize=(10, 10))

    obstacles = [
        (np.array([3.0, 4.0]), 1.0),
        (np.array([6.0, 3.0]), 0.8),
        (np.array([5.0, 7.0]), 1.2),
        (np.array([8.0, 6.0]), 0.9),
    ]

    class MultiCBFController:
        def __init__(self, alpha=1.5):
            self.alpha = alpha

        def safe_control(self, x, u_ref, obstacles):
            u = u_ref.copy()
            for obs_pos, safe_dist in obstacles:
                h = np.sum((x - obs_pos)**2) - safe_dist**2
                grad_h = 2 * (x - obs_pos)
                constraint = grad_h @ u + self.alpha * h
                if constraint < 0:
                    grad_h_norm_sq = np.sum(grad_h**2)
                    if grad_h_norm_sq > 1e-10:
                        lam = -constraint / grad_h_norm_sq
                        u = u - lam * grad_h
            return u

    controller = MultiCBFController(alpha=2.0)
    dt = 0.05
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])

    trajectory = [start.copy()]
    x = start.copy()
    for _ in range(300):
        u_ref = 0.5 * (goal - x) / (np.linalg.norm(goal - x) + 0.1)
        u_safe = controller.safe_control(x, u_ref, obstacles)
        x = x + dt * u_safe
        trajectory.append(x.copy())
        if np.linalg.norm(x - goal) < 0.3:
            break
    trajectory = np.array(trajectory)

    for obs_pos, safe_dist in obstacles:
        circle = plt.Circle(obs_pos, safe_dist, color='red', alpha=0.4)
        ax.add_patch(circle)
        ax.scatter([obs_pos[0]], [obs_pos[1]], c='darkred', s=80, marker='x', zorder=5)

    points = trajectory.reshape(-1, 1, 2)
    from matplotlib.collections import LineCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', linewidth=3)
    lc.set_array(np.linspace(0, 1, len(trajectory) - 1))
    ax.add_collection(lc)

    ax.scatter([start[0]], [start[1]], c='green', s=200, zorder=10, label='起点', marker='o')
    ax.scatter([goal[0]], [goal[1]], c='purple', s=200, marker='*', zorder=10, label='目标')

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('多障碍物CBF避障\n（颜色深浅表示时间推进）', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.colorbar(lc, ax=ax, label='时间')
    plt.tight_layout()
    save_fig('multi_obstacle')

# =============================================================================
# 图5: 可微物理仿真器 - 弹簧质点系统
# =============================================================================
def fig_differentiable_physics():
    """演示可微物理仿真和参数学习"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    true_mass = 1.0
    true_stiffness = 10.0
    true_damping = 0.5
    dt = 0.02

    def simulate(mass, stiffness, damping, x0, v0, controls, dt=0.02):
        states = [(x0, v0)]
        x, v = x0, v0
        for u in controls:
            a = (-stiffness * x - damping * v + u) / mass
            v_new = v + dt * a
            x_new = x + dt * v_new
            states.append((x_new, v_new))
            x, v = x_new, v_new
        return np.array(states)

    np.random.seed(42)
    T = 200
    controls = np.sin(np.linspace(0, 4*np.pi, T)) + 0.5 * np.random.randn(T)
    true_states = simulate(true_mass, true_stiffness, true_damping, 0.0, 0.0, controls, dt)
    noisy_states = true_states + 0.05 * np.random.randn(*true_states.shape)

    ax1 = axes[0, 0]
    t = np.arange(len(true_states)) * dt
    ax1.plot(t, true_states[:, 0], 'b-', linewidth=2, label='位置 x(t)')
    ax1.plot(t, true_states[:, 1], 'r-', linewidth=2, label='速度 v(t)')
    ax1.plot(t[:-1], controls * 0.1, 'g--', alpha=0.5, label='控制 u(t) (缩放)')
    ax1.set_xlabel('时间 (s)', fontsize=12)
    ax1.set_ylabel('状态', fontsize=12)
    ax1.set_title('(a) 弹簧-质点系统轨迹\n(m=1, k=10, c=0.5)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]

    def loss_fn(params, noisy_states, controls):
        mass, stiffness, damping = params
        if mass <= 0 or stiffness <= 0 or damping < 0:
            return 1e6
        pred_states = simulate(mass, stiffness, damping, 0.0, 0.0, controls, dt)
        return np.mean((pred_states - noisy_states)**2)

    params = np.array([2.0, 5.0, 1.0])
    lr = 0.01
    history = {'mass': [params[0]], 'stiffness': [params[1]], 'damping': [params[2]], 'loss': []}

    for i in range(100):
        loss = loss_fn(params, noisy_states, controls)
        history['loss'].append(loss)
        grad = np.zeros(3)
        eps = 1e-4
        for j in range(3):
            params_plus = params.copy()
            params_plus[j] += eps
            grad[j] = (loss_fn(params_plus, noisy_states, controls) - loss) / eps
        params = params - lr * grad
        params = np.clip(params, [0.1, 0.1, 0.0], [10, 50, 5])
        history['mass'].append(params[0])
        history['stiffness'].append(params[1])
        history['damping'].append(params[2])

    iterations = range(len(history['loss']))
    ax2.plot(iterations, history['loss'], 'b-', linewidth=2)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('MSE损失', fontsize=12)
    ax2.set_title('(b) 系统辨识：损失曲线', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    iterations = range(len(history['mass']))
    ax3.plot(iterations, history['mass'], 'b-', linewidth=2, label=f'质量 (真值={true_mass})')
    ax3.plot(iterations, history['stiffness'], 'r-', linewidth=2, label=f'刚度 (真值={true_stiffness})')
    ax3.plot(iterations, history['damping'], 'g-', linewidth=2, label=f'阻尼 (真值={true_damping})')
    ax3.axhline(y=true_mass, color='b', linestyle='--', alpha=0.5)
    ax3.axhline(y=true_stiffness, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=true_damping, color='g', linestyle='--', alpha=0.5)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('参数值', fontsize=12)
    ax3.set_title('(c) 参数收敛过程', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    learned_mass, learned_stiffness, learned_damping = params
    learned_states = simulate(learned_mass, learned_stiffness, learned_damping, 0.0, 0.0, controls, dt)
    ax4.plot(t, true_states[:, 0], 'b-', linewidth=2, label='真实轨迹')
    ax4.plot(t, learned_states[:, 0], 'r--', linewidth=2, label='学习模型预测')
    ax4.scatter(t[::10], noisy_states[::10, 0], c='green', s=20, alpha=0.5, label='带噪声观测')
    ax4.set_xlabel('时间 (s)', fontsize=12)
    ax4.set_ylabel('位置 x(t)', fontsize=12)
    ax4.set_title(f'(d) 模型对比\n学习值: m={learned_mass:.2f}, k={learned_stiffness:.2f}, c={learned_damping:.2f}', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('differentiable_physics')

# =============================================================================
# 图6: 自由能原理示意图
# =============================================================================
def fig_free_energy():
    """自由能原理的可视化解释"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    np.random.seed(42)

    x = np.linspace(-5, 5, 200)
    expected = np.exp(-x**2 / 2)
    expected /= expected.sum()

    obs_positions = [-2, 0, 3]
    surprises = []
    for obs in obs_positions:
        surprise = -np.log(np.exp(-obs**2 / 2) + 1e-10)
        surprises.append(surprise)

    ax1.plot(x, expected * 10, 'b-', linewidth=2, label='预期分布')
    ax1.fill_between(x, 0, expected * 10, alpha=0.3)

    colors = ['green', 'blue', 'red']
    labels = ['低惊奇', '零惊奇', '高惊奇']
    for obs, c, label, surp in zip(obs_positions, colors, labels, surprises):
        ax1.axvline(x=obs, color=c, linestyle='--', linewidth=2)
        ax1.scatter([obs], [np.exp(-obs**2 / 2) * 10], c=c, s=100, zorder=5)
        ax1.annotate(f'{label}\n(S={surp:.2f})', (obs, np.exp(-obs**2 / 2) * 10 + 0.5),
                    ha='center', fontsize=10, color=c)

    ax1.set_xlabel('世界状态', fontsize=12)
    ax1.set_ylabel('概率', fontsize=12)
    ax1.set_title('(a) 惊奇 = -log P(观测)\n智能体最小化惊奇', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    t = np.linspace(0, 10, 100)
    belief_before = 3 * np.exp(-t / 5)
    observation = 0.5 * np.ones_like(t)
    belief_after = belief_before * np.exp(-t / 3) + observation * (1 - np.exp(-t / 3))

    ax2.plot(t, belief_before, 'b--', linewidth=2, label='先验信念', alpha=0.5)
    ax2.plot(t, observation, 'r-', linewidth=2, label='观测')
    ax2.plot(t, belief_after, 'g-', linewidth=2, label='更新后信念')
    ax2.fill_between(t, belief_before, belief_after, alpha=0.2, color='purple', label='感知更新')

    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('状态估计', fontsize=12)
    ax2.set_title('(b) 感知：更新信念\n以匹配观测', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    goal = 2.0
    world_passive = 0.5 * np.ones_like(t)
    action_effect = (goal - 0.5) * (1 - np.exp(-t / 3))
    world_active = world_passive + action_effect

    ax3.plot(t, world_passive, 'b--', linewidth=2, label='世界状态（无动作）', alpha=0.5)
    ax3.axhline(y=goal, color='r', linestyle='-', linewidth=2, label='期望状态（信念）')
    ax3.plot(t, world_active, 'g-', linewidth=2, label='世界状态（有动作）')
    ax3.fill_between(t, world_passive, world_active, alpha=0.2, color='orange', label='动作效果')

    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_ylabel('世界状态', fontsize=12)
    ax3.set_title('(c) 行动：改变世界\n以匹配信念', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('free_energy')

# =============================================================================
# 图7: 相空间轨迹对比
# =============================================================================
def fig_phase_space():
    """展示弹簧质点系统的相空间轨迹"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dt = 0.01
    T = 1000

    def simulate_spring(k, c, x0, v0):
        m = 1.0
        states = []
        x, v = x0, v0
        for _ in range(T):
            a = (-k * x - c * v) / m
            v = v + dt * a
            x = x + dt * v
            states.append([x, v])
        return np.array(states)

    ax1 = axes[0]
    states = simulate_spring(k=4.0, c=0.0, x0=2.0, v0=0.0)
    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5)
    ax1.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='起点')
    ax1.set_xlabel('位置 x', fontsize=12)
    ax1.set_ylabel('速度 v', fontsize=12)
    ax1.set_title('(a) 无阻尼 (c=0)\n能量守恒——闭合轨道', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2 = axes[1]
    states = simulate_spring(k=4.0, c=0.3, x0=2.0, v0=0.0)
    ax2.plot(states[:, 0], states[:, 1], 'r-', linewidth=1.5)
    ax2.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='起点')
    ax2.scatter([0], [0], c='purple', s=100, zorder=5, marker='*', label='平衡点')
    ax2.set_xlabel('位置 x', fontsize=12)
    ax2.set_ylabel('速度 v', fontsize=12)
    ax2.set_title('(b) 欠阻尼 (c=0.3)\n螺旋趋向平衡点', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    states = simulate_spring(k=4.0, c=5.0, x0=2.0, v0=0.0)
    ax3.plot(states[:, 0], states[:, 1], 'g-', linewidth=1.5)
    ax3.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='起点')
    ax3.scatter([0], [0], c='purple', s=100, zorder=5, marker='*', label='平衡点')
    ax3.set_xlabel('位置 x', fontsize=12)
    ax3.set_ylabel('速度 v', fontsize=12)
    ax3.set_title('(c) 过阻尼 (c=5.0)\n直接趋向平衡点', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('phase_space')

# =============================================================================
# 主函数
# =============================================================================
if __name__ == '__main__':
    print("正在生成第16章的所有图表...")
    print("=" * 50)

    fig_paradigm_comparison()
    fig_latent_dynamics()
    fig_cbf_controller()
    fig_multi_obstacle()
    fig_differentiable_physics()
    fig_free_energy()
    fig_phase_space()

    print("=" * 50)
    print(f"所有图表已保存到: {output_dir}/")
    print("完成！")
