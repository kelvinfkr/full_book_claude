#!/usr/bin/env python3

"""

生成chap16的所有可视化图表

包括：世界模型、CBF安全控制器、可微物理仿真

"""

 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False

 

# 设置中文字体（如果可用）

try:

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

except:

    pass

 

import os

 

output_dir = '/home/user/Lecture_16/figs_chap16'

os.makedirs(output_dir, exist_ok=True)

 

# =============================================================================

# 图1: 生成式vs JEPA范式对比

# =============================================================================

def fig_paradigm_comparison():

    """生成两种范式对比图"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

 

    # 左图：生成式模型 - 预测像素

    ax1 = axes[0]

    np.random.seed(42)

    # 模拟视频帧的像素预测

    t = np.linspace(0, 2*np.pi, 100)

    # 真实轨迹（球的抛物线运动）

    x_true = t

    y_true = 5 - 0.5 * (t - np.pi)**2

 

    # 模型预测（有噪声）

    y_pred = y_true + 0.3 * np.random.randn(len(t))

 

    ax1.plot(x_true, y_true, 'b-', linewidth=3, label='True trajectory')

    ax1.scatter(x_true[::5], y_pred[::5], c='r', s=50, alpha=0.6, label='Pixel prediction')

    ax1.fill_between(x_true, y_pred - 0.5, y_pred + 0.5, alpha=0.2, color='red')

    ax1.set_xlabel('Time', fontsize=12)

    ax1.set_ylabel('Position', fontsize=12)

    ax1.set_title('Generative Model: Predict Pixels\n(High-dim, noisy)', fontsize=14)

    ax1.legend()

    ax1.grid(True, alpha=0.3)

 

    # 右图：JEPA - 在潜空间预测

    ax2 = axes[1]

    # 潜空间表示（2D简化）

    z1 = np.cos(t) + 0.05 * np.random.randn(len(t))

    z2 = np.sin(t) + 0.05 * np.random.randn(len(t))

 

    # 预测的潜空间轨迹

    z1_pred = np.cos(t + 0.1)

    z2_pred = np.sin(t + 0.1)

 

    ax2.plot(z1, z2, 'b-', linewidth=3, label='True latent trajectory')

    ax2.plot(z1_pred, z2_pred, 'r--', linewidth=2, label='Predicted trajectory')

    ax2.scatter([z1[0]], [z2[0]], c='green', s=100, zorder=5, marker='o', label='Start')

    ax2.scatter([z1[-1]], [z2[-1]], c='purple', s=100, zorder=5, marker='s', label='End')

    ax2.set_xlabel('Latent dim 1', fontsize=12)

    ax2.set_ylabel('Latent dim 2', fontsize=12)

    ax2.set_title('JEPA: Predict in Latent Space\n(Low-dim, essential)', fontsize=14)

    ax2.legend()

    ax2.grid(True, alpha=0.3)

    ax2.set_aspect('equal')

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/paradigm_comparison.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: paradigm_comparison.png")

 

# =============================================================================

# 图2: 潜空间动力学模型示意图

# =============================================================================

def fig_latent_dynamics():

    """展示潜空间动力学模型如何在"梦境"中规划"""

    fig = plt.figure(figsize=(14, 10))

 

    # 创建多个子图

    # 上方：真实观测空间

    ax1 = fig.add_subplot(2, 2, 1)

    # 左下：潜空间

    ax2 = fig.add_subplot(2, 2, 3)

    # 右边：展开的轨迹和奖励

    ax3 = fig.add_subplot(2, 2, 2)

    ax4 = fig.add_subplot(2, 2, 4)

 

    np.random.seed(123)

 

    # 模拟一个简单的控制问题：从位置0到位置5

    T = 50

    t = np.arange(T)

 

    # 真实观测（高维，有噪声）

    obs_dim = 10

    true_state = 5 * (1 - np.exp(-t / 15))  # 目标位置5

    observations = np.outer(true_state, np.ones(obs_dim)) + 0.5 * np.random.randn(T, obs_dim)

 

    ax1.imshow(observations.T, aspect='auto', cmap='viridis')

    ax1.set_xlabel('Time step', fontsize=11)

    ax1.set_ylabel('Observation dimension', fontsize=11)

    ax1.set_title('(a) High-dim Observations (10D)', fontsize=12)

    ax1.colorbar = plt.colorbar(ax1.images[0], ax=ax1)

 

    # 潜空间表示（2D）

    z1 = true_state / 5  # 归一化到[0,1]

    z2 = 0.3 * np.sin(2 * np.pi * t / T) + 0.1 * np.random.randn(T)

 

    ax2.scatter(z1, z2, c=t, cmap='plasma', s=30)

    ax2.plot(z1, z2, 'k-', alpha=0.3)

    ax2.scatter([z1[0]], [z2[0]], c='green', s=150, marker='o', label='Start', zorder=10)

    ax2.scatter([z1[-1]], [z2[-1]], c='red', s=150, marker='*', label='Goal', zorder=10)

    ax2.set_xlabel('Latent dim 1 (position info)', fontsize=11)

    ax2.set_ylabel('Latent dim 2 (velocity info)', fontsize=11)

    ax2.set_title('(b) Latent Space (2D) - Captures Essentials', fontsize=12)

    ax2.legend()

    ax2.grid(True, alpha=0.3)

 

    # 想象的多条轨迹

    ax3.set_title('(c) "Dreaming": Imagined Trajectories', fontsize=12)

    for i in range(5):

        # 不同的动作序列产生不同轨迹

        noise = 0.3 * np.random.randn(T)

        imagined = true_state + noise.cumsum() * 0.1

        imagined = np.clip(imagined, 0, 7)

        alpha = 0.3 + 0.15 * i

        ax3.plot(t, imagined, alpha=alpha, linewidth=2, label=f'Trajectory {i+1}')

    ax3.axhline(y=5, color='r', linestyle='--', label='Target')

    ax3.set_xlabel('Time step', fontsize=11)

    ax3.set_ylabel('Imagined position', fontsize=11)

    ax3.legend(loc='lower right')

    ax3.grid(True, alpha=0.3)

 

    # 预测奖励

    ax4.set_title('(d) Predicted Rewards for Each Trajectory', fontsize=12)

    for i in range(5):

        rewards = -np.abs(true_state - 5 + 0.5 * np.random.randn(T))

        rewards = np.cumsum(rewards)

        ax4.plot(t, rewards, linewidth=2, label=f'Trajectory {i+1}')

    ax4.set_xlabel('Time step', fontsize=11)

    ax4.set_ylabel('Cumulative reward', fontsize=11)

    ax4.legend(loc='lower left')

    ax4.grid(True, alpha=0.3)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/latent_dynamics.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: latent_dynamics.png")

 

# =============================================================================

# 图3: CBF安全控制器演示

# =============================================================================

def fig_cbf_controller():

    """演示CBF控制器如何保证安全"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

 

    # 障碍物设置

    obstacle_pos = np.array([5.0, 5.0])

    safe_distance = 1.5

 

    # CBF控制器实现

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

                return u_ref, False  # 不需要修正

 

            grad_h_norm_sq = np.sum(grad_h**2)

            if grad_h_norm_sq < 1e-10:

                return u_ref, False

 

            lam = -(grad_h @ u_ref + self.alpha * h) / grad_h_norm_sq

            lam = max(0, lam)

 

            u_safe = u_ref - lam * grad_h

            return u_safe, True  # 需要修正

 

    controller = CBFController(alpha=1.0)

 

    # 模拟轨迹

    dt = 0.1

    T = 100

 

    # 起点和目标

    start = np.array([0.0, 0.0])

    goal = np.array([10.0, 10.0])

 

    # 无安全控制的轨迹

    x_unsafe = [start.copy()]

    x = start.copy()

    for _ in range(T):

        u_ref = 0.3 * (goal - x) / (np.linalg.norm(goal - x) + 0.1)

        x = x + dt * u_ref

        x_unsafe.append(x.copy())

    x_unsafe = np.array(x_unsafe)

 

    # 有CBF安全控制的轨迹

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

 

    # 图1: 无安全控制

    ax1 = axes[0, 0]

    circle = plt.Circle(obstacle_pos, safe_distance, color='red', alpha=0.3)

    ax1.add_patch(circle)

    ax1.plot(x_unsafe[:, 0], x_unsafe[:, 1], 'b-', linewidth=2, label='Trajectory (no CBF)')

    ax1.scatter([start[0]], [start[1]], c='green', s=100, zorder=5, label='Start')

    ax1.scatter([goal[0]], [goal[1]], c='purple', s=100, marker='*', zorder=5, label='Goal')

    ax1.scatter([obstacle_pos[0]], [obstacle_pos[1]], c='red', s=150, marker='x', zorder=5, label='Obstacle')

    ax1.set_xlabel('x', fontsize=12)

    ax1.set_ylabel('y', fontsize=12)

    ax1.set_title('(a) Without CBF: Collision!', fontsize=14)

    ax1.legend()

    ax1.set_xlim(-1, 12)

    ax1.set_ylim(-1, 12)

    ax1.set_aspect('equal')

    ax1.grid(True, alpha=0.3)

 

    # 图2: 有CBF安全控制

    ax2 = axes[0, 1]

    circle = plt.Circle(obstacle_pos, safe_distance, color='red', alpha=0.3)

    ax2.add_patch(circle)

    ax2.plot(x_safe[:, 0], x_safe[:, 1], 'g-', linewidth=2, label='Trajectory (with CBF)')

    ax2.scatter([start[0]], [start[1]], c='green', s=100, zorder=5, label='Start')

    ax2.scatter([goal[0]], [goal[1]], c='purple', s=100, marker='*', zorder=5, label='Goal')

    ax2.scatter([obstacle_pos[0]], [obstacle_pos[1]], c='red', s=150, marker='x', zorder=5, label='Obstacle')

    ax2.set_xlabel('x', fontsize=12)

    ax2.set_ylabel('y', fontsize=12)

    ax2.set_title('(b) With CBF: Safe Avoidance!', fontsize=14)

    ax2.legend()

    ax2.set_xlim(-1, 12)

    ax2.set_ylim(-1, 12)

    ax2.set_aspect('equal')

    ax2.grid(True, alpha=0.3)

 

    # 图3: 障碍函数h(x)随时间变化

    ax3 = axes[1, 0]

    h_unsafe = [np.sum((x - obstacle_pos)**2) - safe_distance**2 for x in x_unsafe]

    h_safe = [np.sum((x - obstacle_pos)**2) - safe_distance**2 for x in x_safe]

    t = np.arange(len(h_unsafe))

 

    ax3.plot(t, h_unsafe, 'b-', linewidth=2, label='h(x) without CBF')

    ax3.plot(t, h_safe, 'g-', linewidth=2, label='h(x) with CBF')

    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Safety boundary (h=0)')

    ax3.fill_between(t, -10, 0, alpha=0.2, color='red', label='Unsafe region (h<0)')

    ax3.set_xlabel('Time step', fontsize=12)

    ax3.set_ylabel('h(x) = ||x - obs||^2 - d^2', fontsize=12)

    ax3.set_title('(c) Barrier Function Value Over Time', fontsize=14)

    ax3.legend()

    ax3.set_ylim(-10, 100)

    ax3.grid(True, alpha=0.3)

 

    # 图4: 控制输入对比

    ax4 = axes[1, 1]

    # 计算控制输入差异

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

    ax4.plot(t, u_ref_list, 'b-', linewidth=2, label='||u_ref|| (desired)')

    ax4.plot(t, u_safe_list, 'g-', linewidth=2, label='||u_safe|| (actual)')

    ax4.fill_between(t, u_safe_list, u_ref_list, where=np.array(corrections),

                     alpha=0.3, color='orange', label='CBF correction active')

    ax4.set_xlabel('Time step', fontsize=12)

    ax4.set_ylabel('Control magnitude', fontsize=12)

    ax4.set_title('(d) Control Input: CBF Modifies Only When Needed', fontsize=14)

    ax4.legend()

    ax4.grid(True, alpha=0.3)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/cbf_controller.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: cbf_controller.png")

 

# =============================================================================

# 图4: 多障碍物避障

# =============================================================================

def fig_multi_obstacle():

    """演示多障碍物环境下的CBF控制"""

    fig, ax = plt.subplots(figsize=(10, 10))

 

    # 多个障碍物

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

 

    # 模拟

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

 

    # 绘制

    for obs_pos, safe_dist in obstacles:

        circle = plt.Circle(obs_pos, safe_dist, color='red', alpha=0.4)

        ax.add_patch(circle)

        ax.scatter([obs_pos[0]], [obs_pos[1]], c='darkred', s=80, marker='x', zorder=5)

 

    # 轨迹颜色渐变

    points = trajectory.reshape(-1, 1, 2)

    from matplotlib.collections import LineCollection

    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis', linewidth=3)

    lc.set_array(np.linspace(0, 1, len(trajectory) - 1))

    ax.add_collection(lc)

 

    ax.scatter([start[0]], [start[1]], c='green', s=200, zorder=10, label='Start', marker='o')

    ax.scatter([goal[0]], [goal[1]], c='purple', s=200, marker='*', zorder=10, label='Goal')

 

    ax.set_xlabel('x', fontsize=14)

    ax.set_ylabel('y', fontsize=14)

    ax.set_title('Multi-Obstacle Avoidance with CBF\n(Color = time progression)', fontsize=16)

    ax.legend(fontsize=12)

    ax.set_xlim(-1, 12)

    ax.set_ylim(-1, 12)

    ax.set_aspect('equal')

    ax.grid(True, alpha=0.3)

 

    plt.colorbar(lc, ax=ax, label='Time')

    plt.tight_layout()

    plt.savefig(f'{output_dir}/multi_obstacle.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: multi_obstacle.png")

 

# =============================================================================

# 图5: 可微物理仿真器 - 弹簧质点系统

# =============================================================================

def fig_differentiable_physics():

    """演示可微物理仿真和参数学习"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

 

    # 真实物理参数

    true_mass = 1.0

    true_stiffness = 10.0

    true_damping = 0.5

    dt = 0.02

 

    # 仿真函数

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

 

    # 生成真实数据

    np.random.seed(42)

    T = 200

    controls = np.sin(np.linspace(0, 4*np.pi, T)) + 0.5 * np.random.randn(T)

    true_states = simulate(true_mass, true_stiffness, true_damping, 0.0, 0.0, controls, dt)

 

    # 添加观测噪声

    noisy_states = true_states + 0.05 * np.random.randn(*true_states.shape)

 

    # 图1: 真实轨迹

    ax1 = axes[0, 0]

    t = np.arange(len(true_states)) * dt

    ax1.plot(t, true_states[:, 0], 'b-', linewidth=2, label='Position x(t)')

    ax1.plot(t, true_states[:, 1], 'r-', linewidth=2, label='Velocity v(t)')

    ax1.plot(t[:-1], controls * 0.1, 'g--', alpha=0.5, label='Control u(t) (scaled)')

    ax1.set_xlabel('Time (s)', fontsize=12)

    ax1.set_ylabel('State', fontsize=12)

    ax1.set_title('(a) Spring-Mass System Trajectory\n(m=1, k=10, c=0.5)', fontsize=14)

    ax1.legend()

    ax1.grid(True, alpha=0.3)

 

    # 图2: 系统辨识 - 梯度下降过程

    ax2 = axes[0, 1]

 

    # 简化的梯度下降系统辨识

    def loss_fn(params, noisy_states, controls):

        mass, stiffness, damping = params

        if mass <= 0 or stiffness <= 0 or damping < 0:

            return 1e6

        pred_states = simulate(mass, stiffness, damping, 0.0, 0.0, controls, dt)

        return np.mean((pred_states - noisy_states)**2)

 

    # 手动梯度下降

    params = np.array([2.0, 5.0, 1.0])  # 初始猜测

    lr = 0.01

    history = {'mass': [params[0]], 'stiffness': [params[1]], 'damping': [params[2]], 'loss': []}

 

    for i in range(100):

        loss = loss_fn(params, noisy_states, controls)

        history['loss'].append(loss)

 

        # 数值梯度

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

    ax2.set_xlabel('Iteration', fontsize=12)

    ax2.set_ylabel('MSE Loss', fontsize=12)

    ax2.set_title('(b) System Identification: Loss Curve', fontsize=14)

    ax2.set_yscale('log')

    ax2.grid(True, alpha=0.3)

 

    # 图3: 参数收敛

    ax3 = axes[1, 0]

    iterations = range(len(history['mass']))

    ax3.plot(iterations, history['mass'], 'b-', linewidth=2, label=f'mass (true={true_mass})')

    ax3.plot(iterations, history['stiffness'], 'r-', linewidth=2, label=f'stiffness (true={true_stiffness})')

    ax3.plot(iterations, history['damping'], 'g-', linewidth=2, label=f'damping (true={true_damping})')

    ax3.axhline(y=true_mass, color='b', linestyle='--', alpha=0.5)

    ax3.axhline(y=true_stiffness, color='r', linestyle='--', alpha=0.5)

    ax3.axhline(y=true_damping, color='g', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Iteration', fontsize=12)

    ax3.set_ylabel('Parameter value', fontsize=12)

    ax3.set_title('(c) Parameter Convergence', fontsize=14)

    ax3.legend()

    ax3.grid(True, alpha=0.3)

 

    # 图4: 学到的模型 vs 真实模型

    ax4 = axes[1, 1]

    learned_mass, learned_stiffness, learned_damping = params

    learned_states = simulate(learned_mass, learned_stiffness, learned_damping, 0.0, 0.0, controls, dt)

 

    ax4.plot(t, true_states[:, 0], 'b-', linewidth=2, label='True trajectory')

    ax4.plot(t, learned_states[:, 0], 'r--', linewidth=2, label='Learned model prediction')

    ax4.scatter(t[::10], noisy_states[::10, 0], c='green', s=20, alpha=0.5, label='Noisy observations')

    ax4.set_xlabel('Time (s)', fontsize=12)

    ax4.set_ylabel('Position x(t)', fontsize=12)

    ax4.set_title(f'(d) Model Comparison\nLearned: m={learned_mass:.2f}, k={learned_stiffness:.2f}, c={learned_damping:.2f}', fontsize=14)

    ax4.legend()

    ax4.grid(True, alpha=0.3)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/differentiable_physics.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: differentiable_physics.png")

 

# =============================================================================

# 图6: 自由能原理示意图

# =============================================================================

def fig_free_energy():

    """自由能原理的可视化解释"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

 

    # 图1: 惊奇最小化

    ax1 = axes[0]

    np.random.seed(42)

 

    # 模拟智能体的预期分布

    x = np.linspace(-5, 5, 200)

    expected = np.exp(-x**2 / 2)  # 预期的世界状态分布

    expected /= expected.sum()

 

    # 不同的观测

    obs_positions = [-2, 0, 3]

    surprises = []

 

    for obs in obs_positions:

        surprise = -np.log(np.exp(-obs**2 / 2) + 1e-10)

        surprises.append(surprise)

 

    ax1.plot(x, expected * 10, 'b-', linewidth=2, label='Expected distribution')

    ax1.fill_between(x, 0, expected * 10, alpha=0.3)

 

    colors = ['green', 'blue', 'red']

    labels = ['Low surprise', 'Zero surprise', 'High surprise']

    for obs, c, label, surp in zip(obs_positions, colors, labels, surprises):

        ax1.axvline(x=obs, color=c, linestyle='--', linewidth=2)

        ax1.scatter([obs], [np.exp(-obs**2 / 2) * 10], c=c, s=100, zorder=5)

        ax1.annotate(f'{label}\n(S={surp:.2f})', (obs, np.exp(-obs**2 / 2) * 10 + 0.5),

                    ha='center', fontsize=10, color=c)

 

    ax1.set_xlabel('World state', fontsize=12)

    ax1.set_ylabel('Probability', fontsize=12)

    ax1.set_title('(a) Surprise = -log P(observation)\nIntelligence minimizes surprise', fontsize=14)

    ax1.legend()

    ax1.grid(True, alpha=0.3)

 

    # 图2: 感知 vs 行动

    ax2 = axes[1]

 

    # 感知：更新内部模型

    t = np.linspace(0, 10, 100)

    belief_before = 3 * np.exp(-t / 5)  # 初始信念

    observation = 0.5 * np.ones_like(t)  # 真实观测

    belief_after = belief_before * np.exp(-t / 3) + observation * (1 - np.exp(-t / 3))

 

    ax2.plot(t, belief_before, 'b--', linewidth=2, label='Prior belief', alpha=0.5)

    ax2.plot(t, observation, 'r-', linewidth=2, label='Observation')

    ax2.plot(t, belief_after, 'g-', linewidth=2, label='Updated belief')

    ax2.fill_between(t, belief_before, belief_after, alpha=0.2, color='purple', label='Perception update')

 

    ax2.set_xlabel('Time', fontsize=12)

    ax2.set_ylabel('State estimate', fontsize=12)

    ax2.set_title('(b) Perception: Update beliefs\nto match observations', fontsize=14)

    ax2.legend()

    ax2.grid(True, alpha=0.3)

 

    # 图3: 行动改变世界

    ax3 = axes[2]

 

    # 行动：改变世界使其符合预期

    goal = 2.0

    world_passive = 0.5 * np.ones_like(t)  # 被动世界状态

    action_effect = (goal - 0.5) * (1 - np.exp(-t / 3))  # 行动的效果

    world_active = world_passive + action_effect

 

    ax3.plot(t, world_passive, 'b--', linewidth=2, label='World (no action)', alpha=0.5)

    ax3.axhline(y=goal, color='r', linestyle='-', linewidth=2, label='Desired state (belief)')

    ax3.plot(t, world_active, 'g-', linewidth=2, label='World (with action)')

    ax3.fill_between(t, world_passive, world_active, alpha=0.2, color='orange', label='Action effect')

 

    ax3.set_xlabel('Time', fontsize=12)

    ax3.set_ylabel('World state', fontsize=12)

    ax3.set_title('(c) Action: Change world\nto match beliefs', fontsize=14)

    ax3.legend()

    ax3.grid(True, alpha=0.3)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/free_energy.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: free_energy.png")

 

# =============================================================================

# 图7: 约束优化统一视角

# =============================================================================

def fig_unified_view():

    """展示约束优化如何统一不同领域"""

    fig = plt.figure(figsize=(14, 10))

 

    # 创建一个大的概念图

    ax = fig.add_subplot(111)

    ax.set_xlim(0, 14)

    ax.set_ylim(0, 10)

    ax.axis('off')

 

    # 中心：约束优化

    center = plt.Circle((7, 5), 1.5, color='gold', alpha=0.8)

    ax.add_patch(center)

    ax.text(7, 5, 'Constrained\nOptimization\nmin f(x)\ns.t. g(x)=0\nh(x)>=0',

            ha='center', va='center', fontsize=11, fontweight='bold')

 

    # 周围的应用领域

    applications = [

        (2, 8, 'Lagrangian\nMechanics', 'blue', 'Geometric\nConstraints'),

        (7, 9, 'Optimal\nControl', 'green', 'Dynamics\nConstraints'),

        (12, 8, 'PINN', 'red', 'PDE\nConstraints'),

        (12, 2, 'Neural\nOperators', 'purple', 'Operator\nConstraints'),

        (7, 1, 'Multi-Agent\nGames', 'orange', 'Equilibrium\nConstraints'),

        (2, 2, 'World\nModels', 'cyan', 'Physical+Safety\nConstraints'),

    ]

 

    for x, y, name, color, constraint in applications:

        circle = plt.Circle((x, y), 1.2, color=color, alpha=0.5)

        ax.add_patch(circle)

        ax.text(x, y + 0.2, name, ha='center', va='center', fontsize=10, fontweight='bold')

        ax.text(x, y - 0.5, constraint, ha='center', va='center', fontsize=8, color='gray')

 

        # 连线到中心

        ax.annotate('', xy=(7 + 1.5 * (x - 7) / np.sqrt((x-7)**2 + (y-5)**2 + 0.1),

                           5 + 1.5 * (y - 5) / np.sqrt((x-7)**2 + (y-5)**2 + 0.1)),

                   xytext=(x - 1.2 * (x - 7) / np.sqrt((x-7)**2 + (y-5)**2 + 0.1),

                          y - 1.2 * (y - 5) / np.sqrt((x-7)**2 + (y-5)**2 + 0.1)),

                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

 

    ax.set_title('Unified View: Everything is Constrained Optimization!', fontsize=16, fontweight='bold', y=1.02)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/unified_view.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: unified_view.png")

 

# =============================================================================

# 图8: 相空间轨迹对比

# =============================================================================

def fig_phase_space():

    """展示弹簧质点系统的相空间轨迹"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

 

    dt = 0.01

    T = 1000

 

    def simulate_spring(k, c, x0, v0):

        """模拟弹簧质点系统: m*a = -k*x - c*v"""

        m = 1.0

        states = []

        x, v = x0, v0

        for _ in range(T):

            a = (-k * x - c * v) / m

            v = v + dt * a

            x = x + dt * v

            states.append([x, v])

        return np.array(states)

 

    # 无阻尼

    ax1 = axes[0]

    states = simulate_spring(k=4.0, c=0.0, x0=2.0, v0=0.0)

    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5)

    ax1.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='Start')

    ax1.set_xlabel('Position x', fontsize=12)

    ax1.set_ylabel('Velocity v', fontsize=12)

    ax1.set_title('(a) Undamped (c=0)\nEnergy conserved - closed orbit', fontsize=14)

    ax1.legend()

    ax1.grid(True, alpha=0.3)

    ax1.set_aspect('equal')

 

    # 欠阻尼

    ax2 = axes[1]

    states = simulate_spring(k=4.0, c=0.3, x0=2.0, v0=0.0)

    ax2.plot(states[:, 0], states[:, 1], 'r-', linewidth=1.5)

    ax2.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='Start')

    ax2.scatter([0], [0], c='purple', s=100, zorder=5, marker='*', label='Equilibrium')

    ax2.set_xlabel('Position x', fontsize=12)

    ax2.set_ylabel('Velocity v', fontsize=12)

    ax2.set_title('(a) Underdamped (c=0.3)\nSpiral to equilibrium', fontsize=14)

    ax2.legend()

    ax2.grid(True, alpha=0.3)

 

    # 过阻尼

    ax3 = axes[2]

    states = simulate_spring(k=4.0, c=5.0, x0=2.0, v0=0.0)

    ax3.plot(states[:, 0], states[:, 1], 'g-', linewidth=1.5)

    ax3.scatter([states[0, 0]], [states[0, 1]], c='green', s=100, zorder=5, label='Start')

    ax3.scatter([0], [0], c='purple', s=100, zorder=5, marker='*', label='Equilibrium')

    ax3.set_xlabel('Position x', fontsize=12)

    ax3.set_ylabel('Velocity v', fontsize=12)

    ax3.set_title('(c) Overdamped (c=5.0)\nDirect approach', fontsize=14)

    ax3.legend()

    ax3.grid(True, alpha=0.3)

 

    plt.tight_layout()

    plt.savefig(f'{output_dir}/phase_space.png', dpi=150, bbox_inches='tight')

    plt.close()

    print("Generated: phase_space.png")

 

# =============================================================================

# 主函数

# =============================================================================

if __name__ == '__main__':

    print("Generating figures for Chapter 16...")

    print("=" * 50)

 

    fig_paradigm_comparison()

    fig_latent_dynamics()

    fig_cbf_controller()

    fig_multi_obstacle()

    fig_differentiable_physics()

    fig_free_energy()

    fig_unified_view()

    fig_phase_space()

 

    print("=" * 50)

    print(f"All figures saved to: {output_dir}/")

    print("Done!")