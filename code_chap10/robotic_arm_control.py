"""
第10章扩展：机械臂控制示例
演示连续控制任务的强化学习

内容：
1. 2-link平面机械臂正/逆运动学
2. 简单的PD控制器基线
3. CartPole的SB3训练演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 第1部分：2-link平面机械臂
# =============================================================================

class TwoLinkArm:
    """二连杆平面机械臂"""
    def __init__(self, L1=1.0, L2=0.8, m1=1.0, m2=0.8):
        self.L1 = L1  # 第一杆长度
        self.L2 = L2  # 第二杆长度
        self.m1 = m1  # 第一杆质量
        self.m2 = m2  # 第二杆质量
        self.g = 9.81  # 重力加速度

        # 状态：[θ1, θ2, θ1_dot, θ2_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

    def forward_kinematics(self, theta1, theta2):
        """正运动学：关节角度 -> 末端位置"""
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        return (x1, y1), (x2, y2)

    def inverse_kinematics(self, x_target, y_target):
        """逆运动学：末端位置 -> 关节角度"""
        r = np.sqrt(x_target**2 + y_target**2)

        # 检查可达性
        if r > self.L1 + self.L2 or r < abs(self.L1 - self.L2):
            return None, None

        # 余弦定理求θ2
        cos_theta2 = (r**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)  # 取肘部朝上的解

        # 求θ1
        beta = np.arctan2(y_target, x_target)
        psi = np.arctan2(self.L2 * np.sin(theta2),
                        self.L1 + self.L2 * np.cos(theta2))
        theta1 = beta - psi

        return theta1, theta2

    def jacobian(self, theta1, theta2):
        """雅可比矩阵：dX/dθ"""
        J = np.array([
            [-self.L1*np.sin(theta1) - self.L2*np.sin(theta1+theta2),
             -self.L2*np.sin(theta1+theta2)],
            [self.L1*np.cos(theta1) + self.L2*np.cos(theta1+theta2),
             self.L2*np.cos(theta1+theta2)]
        ])
        return J

    def dynamics(self, state, tau):
        """动力学方程（简化版，忽略重力）"""
        theta1, theta2, dtheta1, dtheta2 = state

        # 质量矩阵 M(θ)
        M11 = (self.m1 + self.m2) * self.L1**2 + self.m2 * self.L2**2 + \
              2 * self.m2 * self.L1 * self.L2 * np.cos(theta2)
        M12 = self.m2 * self.L2**2 + self.m2 * self.L1 * self.L2 * np.cos(theta2)
        M22 = self.m2 * self.L2**2
        M = np.array([[M11, M12], [M12, M22]])

        # 科里奥利和离心力 C(θ, θ_dot)
        h = self.m2 * self.L1 * self.L2 * np.sin(theta2)
        C = np.array([[-h * dtheta2, -h * (dtheta1 + dtheta2)],
                      [h * dtheta1, 0]])

        # 求解加速度：M * ddθ + C * dθ = τ
        dtheta = np.array([dtheta1, dtheta2])
        ddtheta = np.linalg.solve(M, tau - C @ dtheta)

        return np.array([dtheta1, dtheta2, ddtheta[0], ddtheta[1]])

    def step(self, tau, dt=0.01):
        """前向仿真一步"""
        # RK4积分
        k1 = self.dynamics(self.state, tau)
        k2 = self.dynamics(self.state + 0.5*dt*k1, tau)
        k3 = self.dynamics(self.state + 0.5*dt*k2, tau)
        k4 = self.dynamics(self.state + dt*k3, tau)
        self.state = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return self.state.copy()

# =============================================================================
# 第2部分：PD控制器
# =============================================================================

class PDController:
    """关节空间PD控制器"""
    def __init__(self, Kp=50.0, Kd=10.0):
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, theta_current, theta_target, dtheta_current):
        """计算控制力矩"""
        error = theta_target - theta_current
        tau = self.Kp * error - self.Kd * dtheta_current
        return tau

# =============================================================================
# 第3部分：可视化
# =============================================================================

def plot_arm_kinematics(save_path):
    """绘制机械臂运动学示意图"""
    arm = TwoLinkArm()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. 正运动学
    ax1 = axes[0]
    theta1_range = np.linspace(-np.pi/2, np.pi/2, 5)
    theta2_range = np.linspace(-np.pi/2, np.pi/2, 5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(theta1_range)))

    for i, theta1 in enumerate(theta1_range):
        for theta2 in theta2_range:
            (x1, y1), (x2, y2) = arm.forward_kinematics(theta1, theta2)
            ax1.plot([0, x1, x2], [0, y1, y2], 'o-', color=colors[i],
                    alpha=0.5, linewidth=1.5, markersize=3)

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('Forward Kinematics: Workspace', fontsize=12)
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # 绘制工作空间边界
    theta = np.linspace(0, 2*np.pi, 100)
    r_outer = arm.L1 + arm.L2
    r_inner = abs(arm.L1 - arm.L2)
    ax1.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 'r--',
            linewidth=1, label='Workspace boundary')
    ax1.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 'r--', linewidth=1)
    ax1.legend(fontsize=9, loc='upper right')

    # 2. 逆运动学
    ax2 = axes[1]
    target_points = [(1.2, 0.5), (0.8, 1.0), (1.5, 0.2), (0.5, 0.8)]

    for i, (x_t, y_t) in enumerate(target_points):
        theta1, theta2 = arm.inverse_kinematics(x_t, y_t)
        if theta1 is not None:
            (x1, y1), (x2, y2) = arm.forward_kinematics(theta1, theta2)
            color = plt.cm.Set1(i/len(target_points))
            ax2.plot([0, x1, x2], [0, y1, y2], 'o-', color=color,
                    linewidth=2, markersize=6, label=f'Target ({x_t}, {y_t})')
            ax2.plot(x_t, y_t, 's', color=color, markersize=10)

    ax2.set_xlim(-0.5, 2)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (m)', fontsize=11)
    ax2.set_ylabel('y (m)', fontsize=11)
    ax2.set_title('Inverse Kinematics: Reaching Targets', fontsize=12)
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.axvline(x=0, color='gray', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='upper left')

    # 3. 雅可比矩阵可视化
    ax3 = axes[2]

    # 固定一个位姿，绘制雅可比对应的速度椭圆
    theta1, theta2 = np.pi/4, np.pi/6
    (x1, y1), (x2, y2) = arm.forward_kinematics(theta1, theta2)
    J = arm.jacobian(theta1, theta2)

    # 绘制机械臂
    ax3.plot([0, x1, x2], [0, y1, y2], 'ko-', linewidth=3, markersize=8)
    ax3.plot(0, 0, 'ko', markersize=12)  # 基座

    # 绘制速度椭圆（通过SVD）
    U, s, Vt = np.linalg.svd(J)
    angle = np.arctan2(U[1, 0], U[0, 0])

    ellipse_theta = np.linspace(0, 2*np.pi, 100)
    scale = 0.3
    ellipse_x = scale * s[0] * np.cos(ellipse_theta)
    ellipse_y = scale * s[1] * np.sin(ellipse_theta)

    # 旋转椭圆
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    ellipse_pts = R @ np.array([ellipse_x, ellipse_y])

    ax3.plot(x2 + ellipse_pts[0], y2 + ellipse_pts[1], 'b-', linewidth=2,
            label='Manipulability ellipse')
    ax3.fill(x2 + ellipse_pts[0], y2 + ellipse_pts[1], alpha=0.2, color='blue')

    # 绘制主方向
    for i in range(2):
        direction = U[:, i] * s[i] * scale
        ax3.arrow(x2, y2, direction[0], direction[1],
                 head_width=0.05, head_length=0.03, fc='red', ec='red')

    ax3.set_xlim(-0.5, 2)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x (m)', fontsize=11)
    ax3.set_ylabel('y (m)', fontsize=11)
    ax3.set_title('Jacobian: Manipulability Ellipse', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pd_control_trajectory(save_path):
    """绘制PD控制轨迹跟踪"""
    arm = TwoLinkArm()
    controller = PDController(Kp=100.0, Kd=20.0)

    # 初始状态
    arm.state = np.array([0.0, 0.0, 0.0, 0.0])

    # 目标轨迹：画圆
    t_total = 5.0
    dt = 0.01
    t = np.arange(0, t_total, dt)

    # 目标圆心和半径
    cx, cy = 1.0, 0.5
    radius = 0.3

    # 记录数据
    history = {'t': [], 'x': [], 'y': [], 'x_target': [], 'y_target': [],
               'theta1': [], 'theta2': [], 'tau1': [], 'tau2': []}

    for ti in t:
        # 目标位置（圆形轨迹）
        x_target = cx + radius * np.cos(2 * np.pi * ti / t_total)
        y_target = cy + radius * np.sin(2 * np.pi * ti / t_total)

        # 逆运动学得到目标角度
        theta1_target, theta2_target = arm.inverse_kinematics(x_target, y_target)
        if theta1_target is None:
            continue

        # PD控制
        theta_current = arm.state[:2]
        dtheta_current = arm.state[2:]
        theta_target = np.array([theta1_target, theta2_target])

        tau = controller.compute(theta_current, theta_target, dtheta_current)
        tau = np.clip(tau, -50, 50)  # 限幅

        # 仿真一步
        arm.step(tau, dt)

        # 记录
        (x1, y1), (x2, y2) = arm.forward_kinematics(arm.state[0], arm.state[1])
        history['t'].append(ti)
        history['x'].append(x2)
        history['y'].append(y2)
        history['x_target'].append(x_target)
        history['y_target'].append(y_target)
        history['theta1'].append(arm.state[0])
        history['theta2'].append(arm.state[1])
        history['tau1'].append(tau[0])
        history['tau2'].append(tau[1])

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 末端轨迹
    ax1 = axes[0, 0]
    ax1.plot(history['x_target'], history['y_target'], 'r--', linewidth=2,
            label='Target trajectory')
    ax1.plot(history['x'], history['y'], 'b-', linewidth=1.5,
            label='Actual trajectory')
    ax1.plot(history['x'][0], history['y'][0], 'go', markersize=10, label='Start')
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('End-effector Trajectory', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. 位置误差
    ax2 = axes[0, 1]
    error_x = np.array(history['x']) - np.array(history['x_target'])
    error_y = np.array(history['y']) - np.array(history['y_target'])
    error = np.sqrt(error_x**2 + error_y**2)
    ax2.plot(history['t'], error * 1000, 'b-', linewidth=1.5)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position error (mm)', fontsize=11)
    ax2.set_title('Tracking Error', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. 关节角度
    ax3 = axes[1, 0]
    ax3.plot(history['t'], np.rad2deg(history['theta1']), 'b-',
            linewidth=1.5, label=r'$\theta_1$')
    ax3.plot(history['t'], np.rad2deg(history['theta2']), 'r-',
            linewidth=1.5, label=r'$\theta_2$')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Joint angle (deg)', fontsize=11)
    ax3.set_title('Joint Angles', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 控制力矩
    ax4 = axes[1, 1]
    ax4.plot(history['t'], history['tau1'], 'b-', linewidth=1.5, label=r'$\tau_1$')
    ax4.plot(history['t'], history['tau2'], 'r-', linewidth=1.5, label=r'$\tau_2$')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Torque (N·m)', fontsize=11)
    ax4.set_title('Control Torques', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_cartpole_demo(save_path):
    """绘制CartPole环境示意图和训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. CartPole环境示意图
    ax1 = axes[0]
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')

    # 轨道
    ax1.plot([-2.5, 2.5], [0, 0], 'k-', linewidth=3)

    # 小车
    cart_x, cart_y = 0, 0.15
    cart_width, cart_height = 0.6, 0.3
    cart = plt.Rectangle((cart_x - cart_width/2, cart_y - cart_height/2),
                         cart_width, cart_height, facecolor='blue', edgecolor='black')
    ax1.add_patch(cart)

    # 轮子
    wheel1 = Circle((cart_x - 0.2, 0), 0.08, facecolor='gray', edgecolor='black')
    wheel2 = Circle((cart_x + 0.2, 0), 0.08, facecolor='gray', edgecolor='black')
    ax1.add_patch(wheel1)
    ax1.add_patch(wheel2)

    # 杆（倾斜）
    pole_angle = 15 * np.pi / 180  # 15度倾斜
    pole_length = 1.5
    pole_end_x = cart_x + pole_length * np.sin(pole_angle)
    pole_end_y = cart_y + cart_height/2 + pole_length * np.cos(pole_angle)
    ax1.plot([cart_x, pole_end_x], [cart_y + cart_height/2, pole_end_y],
            'orange', linewidth=8, solid_capstyle='round')
    ax1.plot(pole_end_x, pole_end_y, 'ro', markersize=12)

    # 标注
    ax1.annotate('', xy=(1.5, 0.15), xytext=(0.5, 0.15),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(1.0, 0.4, 'Action: Push left/right', fontsize=10, color='green')

    ax1.annotate('', xy=(pole_end_x + 0.3, pole_end_y - 0.2),
                xytext=(pole_end_x, pole_end_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(pole_end_x + 0.4, pole_end_y - 0.1, r'$\theta$', fontsize=14, color='red')

    # 状态标注
    ax1.text(-2.5, 2.2, 'State: [x, dx/dt, θ, dθ/dt]', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.text(-2.5, 1.8, 'Reward: +1 for each step alive', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax1.set_title('CartPole Environment', fontsize=13)
    ax1.axis('off')

    # 2. 模拟训练曲线
    ax2 = axes[1]

    # 模拟不同算法的学习曲线
    np.random.seed(42)
    episodes = np.arange(0, 501, 10)

    # DQN曲线
    dqn_base = 195 * (1 - np.exp(-episodes / 150))
    dqn_noise = np.random.randn(len(episodes)) * 20
    dqn_reward = np.clip(dqn_base + dqn_noise, 10, 200)

    # PPO曲线（更快收敛）
    ppo_base = 200 * (1 - np.exp(-episodes / 80))
    ppo_noise = np.random.randn(len(episodes)) * 15
    ppo_reward = np.clip(ppo_base + ppo_noise, 10, 200)

    # Random曲线
    random_reward = 20 + np.random.randn(len(episodes)) * 5

    ax2.plot(episodes, dqn_reward, 'b-', linewidth=2, label='DQN', alpha=0.8)
    ax2.plot(episodes, ppo_reward, 'g-', linewidth=2, label='PPO', alpha=0.8)
    ax2.plot(episodes, random_reward, 'r--', linewidth=1.5, label='Random', alpha=0.6)
    ax2.axhline(y=195, color='gray', linestyle=':', linewidth=1, label='Solved (195)')

    ax2.fill_between(episodes, dqn_reward - 10, dqn_reward + 10, alpha=0.1, color='blue')
    ax2.fill_between(episodes, ppo_reward - 10, ppo_reward + 10, alpha=0.1, color='green')

    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Total Reward', fontsize=11)
    ax2.set_title('CartPole Training Curves (Simulated)', fontsize=12)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 220)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_rl_control_comparison(save_path):
    """对比传统控制与强化学习"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. 传统控制流程
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('Traditional Control Pipeline', fontsize=13, fontweight='bold')

    # 流程框
    boxes = [
        (1, 6, 'System\nModeling'),
        (4, 6, 'Controller\nDesign'),
        (7, 6, 'Stability\nAnalysis'),
        (1, 3, 'Parameter\nTuning'),
        (4, 3, 'Implementation'),
        (7, 3, 'Testing'),
    ]

    for x, y, text in boxes:
        rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2,
                             facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # 箭头
    arrows = [((2.2, 6), (3.2, 6)), ((5.6, 6), (6.2, 6)),
              ((1, 5.4), (1, 3.6)), ((4, 5.4), (4, 3.6)), ((7, 5.4), (7, 3.6)),
              ((2.2, 3), (3.2, 3)), ((5.6, 3), (6.2, 3))]
    for (x1, y1), (x2, y2) in arrows:
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.text(5, 1, 'Requires: Expert knowledge, accurate model',
            fontsize=10, ha='center', style='italic')

    # 2. 强化学习流程
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('Reinforcement Learning Pipeline', fontsize=13, fontweight='bold')

    # 简化流程
    rl_boxes = [
        (2, 6, 'Define\nReward'),
        (5, 6, 'Agent\n(Neural Net)'),
        (8, 6, 'Environment'),
        (5, 3, 'Training Loop'),
    ]

    colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
    for (x, y, text), color in zip(rl_boxes, colors):
        rect = plt.Rectangle((x-0.9, y-0.6), 1.8, 1.2,
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # 循环箭头
    ax2.annotate('', xy=(6.1, 6), xytext=(5.9, 6),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                               connectionstyle='arc3,rad=0.5'))
    ax2.annotate('Action', xy=(6.5, 6.8), fontsize=9, color='blue')

    ax2.annotate('', xy=(5.1, 6), xytext=(6.9, 6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                               connectionstyle='arc3,rad=-0.5'))
    ax2.annotate('State, Reward', xy=(5.5, 5.2), fontsize=9, color='red')

    # 到训练循环的箭头
    ax2.annotate('', xy=(5, 4.2), xytext=(5, 5.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax2.text(5, 1, 'Requires: Reward design, compute, environment',
            fontsize=10, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap10"

    print("=" * 60)
    print("Robotic Arm Control Demo")
    print("=" * 60)

    # 1. 机械臂运动学
    print("\n[1] Plotting arm kinematics...")
    plot_arm_kinematics(f"{output_dir}/arm_kinematics.pdf")

    # 2. PD控制轨迹
    print("\n[2] Plotting PD control trajectory...")
    plot_pd_control_trajectory(f"{output_dir}/pd_control_trajectory.pdf")

    # 3. CartPole演示
    print("\n[3] Plotting CartPole demo...")
    plot_cartpole_demo(f"{output_dir}/cartpole_demo.pdf")

    # 4. 控制方法对比
    print("\n[4] Plotting control comparison...")
    plot_rl_control_comparison(f"{output_dir}/rl_control_comparison.pdf")

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)
