#!/usr/bin/env python3
"""
第7章：机器人运动学与动力学演示

本代码演示机器人学中的核心数学概念，包括：
1. SVD投影到SO(3)：如何把任意矩阵"修正"成合法的旋转矩阵
2. 旋转矩阵的正确更新方式：为什么简单的欧拉积分会破坏约束
3. 平面双连杆机械臂：正运动学（关节角→末端位置）和逆运动学（末端位置→关节角）
4. 雅可比矩阵与奇异性：什么时候机械臂会"卡住"

对应书中内容：
- 图7.1：SVD投影可视化
- 图7.7-7.10：双连杆机械臂的运动学分析
- 图7.14：力传递与雅可比转置（τ = J^T F）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation

# 导入中文字体配置（确保matplotlib能正确显示中文）
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第7章：机器人运动学与动力学演示")
print("="*60)

# ==================== 实验1：SVD投影到SO(3) ====================
#
# 背景知识：
# - SO(3) 是所有 3×3 旋转矩阵的集合
# - 旋转矩阵满足两个条件：R^T R = I（正交）且 det(R) = +1（不含镜像）
# - 实际应用中（如传感器融合），矩阵可能因噪声而偏离SO(3)
# - 需要一种方法把"几乎是旋转矩阵"的矩阵"修正"成真正的旋转矩阵
#
# 数学原理：
# - SVD分解：A = U Σ V^T，其中 U, V 是正交矩阵
# - 取 R = U V^T，这是 Frobenius 范数意义下最接近 A 的正交矩阵
# - 若 det(R) = -1，翻转 V 的最后一行使 det(R) = +1
#
print("\n" + "="*60)
print("实验1：用SVD把矩阵投影到SO(3)")
print("="*60)

# 构造一个"有噪声"的矩阵（接近旋转矩阵但不完全满足约束）
A = np.array([
    [0.9, -0.5, 0.1],    # 每行不是单位向量
    [0.4, 0.85, -0.3],   # 行之间不完全正交
    [-0.1, 0.25, 0.95]   # 整体结构接近旋转矩阵
])

# 验证 A 不满足旋转矩阵的条件
print("\nA 不是旋转矩阵：")
print("A^T @ A =\n", np.round(A.T @ A, 3))  # 应该等于单位矩阵 I，但不是
print("det(A) =", np.round(np.linalg.det(A), 3))  # 应该等于 1，但不是

# 用 SVD 投影到 SO(3)
# 这是最小化 ||A - R||_F 的最优解，其中 R ∈ SO(3)
U, S, Vt = np.linalg.svd(A)  # A = U @ diag(S) @ Vt
R = U @ Vt  # 丢弃奇异值，保留正交结构

# 确保 det = +1（排除镜像/反射）
# 如果 det = -1，说明 R 是"反射"而非"旋转"
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1  # 翻转最后一行
    R = U @ Vt

# 验证 R 现在是合法的旋转矩阵
print("\nR 是旋转矩阵：")
print("R^T @ R =\n", np.round(R.T @ R, 3))  # 现在等于单位矩阵 I
print("det(R) =", np.round(np.linalg.det(R), 3))  # 现在等于 1

# ==================== 实验2：旋转矩阵的正确更新 ====================
#
# 问题背景：
# - 在机器人仿真或姿态估计中，需要根据角速度更新旋转矩阵
# - 直觉做法：R_new = R_old + dt * dR/dt（欧拉积分）
# - 问题：这会破坏 R^T R = I 的约束！
#
# 解决方案：指数映射（Rodrigues公式）
# - 角速度 ω 对应李代数 so(3) 中的反对称矩阵 [ω]×
# - 指数映射 exp([ω]× dt) 给出增量旋转 ΔR
# - 正确更新：R_new = R_old @ ΔR（矩阵乘法，不是加法！）
# - 这保证 R_new 仍然在 SO(3) 中
#
print("\n" + "="*60)
print("实验2：旋转矩阵的正确更新方式")
print("="*60)

def skew(w):
    """
    构造反对称矩阵（又称叉乘矩阵）

    输入：3维向量 w = [w1, w2, w3]
    输出：3×3反对称矩阵 [w]×，满足 [w]× @ v = w × v（叉乘）

    数学性质：
    - [w]×^T = -[w]×（反对称）
    - 对任意向量 v：[w]× v = w × v
    - 是李代数 so(3) 的元素
    """
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def exp_so3(omega):
    """
    指数映射：李代数 so(3) -> 李群 SO(3)

    使用 Rodrigues 公式计算：
    exp([ω]×) = I + sin(θ) K + (1-cos(θ)) K²

    其中：
    - θ = ||ω|| 是旋转角度
    - K = [ω/θ]× 是单位轴的反对称矩阵

    物理意义：
    - 绕轴 ω/||ω|| 旋转角度 ||ω||
    - 这是最短路径旋转（测地线）
    """
    theta = np.linalg.norm(omega)  # 旋转角度
    if theta < 1e-10:  # 角度太小，返回单位矩阵（避免除零）
        return np.eye(3)
    w_hat = omega / theta  # 单位旋转轴
    K = skew(w_hat)  # 轴对应的反对称矩阵
    # Rodrigues 公式
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

# 设置初始条件
R0 = np.eye(3)  # 初始姿态：单位矩阵（无旋转）
omega = np.array([0.1, 0.2, 0.3])  # 角速度向量 (rad/s)
dt = 0.1  # 时间步长

# ===== 错误方法：简单欧拉积分 =====
# 这是直觉但错误的做法：R_new = R_old + dt * R_dot
# 其中 R_dot = R @ [ω]×（角速度导致的旋转矩阵变化率）
R_wrong = R0 + dt * R0 @ skew(omega)

print("\n错误的欧拉更新后：")
print("R^T @ R =\n", np.round(R_wrong.T @ R_wrong, 4))  # 应该是 I，但不是！
print("det(R) =", np.round(np.linalg.det(R_wrong), 4))  # 应该是 1，但不是！
print("结论：不再是旋转矩阵！")

# ===== 正确方法：指数映射 =====
# 计算增量旋转：ΔR = exp([ω dt]×)
# 然后用矩阵乘法更新：R_new = R_old @ ΔR
delta_R = exp_so3(omega * dt)  # 小角度旋转
R_correct = R0 @ delta_R  # 复合旋转（右乘：body-fixed frame）

print("\n正确的指数映射更新后：")
print("R^T @ R =\n", np.round(R_correct.T @ R_correct, 4))  # 精确等于 I
print("det(R) =", np.round(np.linalg.det(R_correct), 4))  # 精确等于 1
print("结论：仍然是旋转矩阵！")

# ==================== 实验3：平面双连杆机械臂 ====================
#
# 这是机器人学的经典模型，用于演示：
# - 正运动学：给定关节角，求末端位置
# - 逆运动学：给定目标位置，求所需关节角
# - 雅可比矩阵：末端速度与关节速度的关系
# - 奇异性：什么时候机械臂会"卡住"
#
# 几何设置：
#        (第二关节)
#           O-----------* (末端)
#          /  连杆2, 长度L2
#         / 角度q2（相对于连杆1）
#   -----O (第一关节/基座)
#    连杆1, 长度L1
#    角度q1（相对于水平）
#
print("\n" + "="*60)
print("实验3：平面双连杆机械臂正/逆运动学")
print("="*60)

class TwoLinkArm:
    """
    平面双连杆机械臂模型

    这是机器人学中最简单但最具教育意义的模型。
    它展示了正/逆运动学、雅可比矩阵、奇异性等核心概念。
    """

    def __init__(self, L1=1.0, L2=1.0):
        """
        初始化机械臂参数

        参数:
        - L1: 第一连杆长度（从基座到第二关节）
        - L2: 第二连杆长度（从第二关节到末端）
        """
        self.L1 = L1
        self.L2 = L2

    def forward_kinematics(self, q1, q2):
        """
        正运动学：关节角 -> 末端位置

        给定两个关节角 (q1, q2)，计算末端在世界坐标系中的位置。

        公式推导（几何法）：
        - 第一关节位置：(0, 0)
        - 第二关节位置：(L1*cos(q1), L1*sin(q1))
        - 末端位置：第二关节 + 连杆2向量
          x = L1*cos(q1) + L2*cos(q1+q2)
          y = L1*sin(q1) + L2*sin(q1+q2)

        注意：q1是绝对角度，q2是相对角度（相对于连杆1的方向）
        """
        x = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        y = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        return np.array([x, y])

    def jacobian(self, q1, q2):
        """
        计算雅可比矩阵 J = ∂x/∂q

        雅可比矩阵描述了关节速度到末端速度的映射：
            ẋ = J(q) q̇

        J 是 2×2 矩阵：
            J = [∂x/∂q1  ∂x/∂q2]
                [∂y/∂q1  ∂y/∂q2]

        通过对正运动学公式求偏导得到：
            ∂x/∂q1 = -L1*sin(q1) - L2*sin(q1+q2)
            ∂x/∂q2 = -L2*sin(q1+q2)
            ∂y/∂q1 = L1*cos(q1) + L2*cos(q1+q2)
            ∂y/∂q2 = L2*cos(q1+q2)
        """
        J = np.array([
            [-self.L1*np.sin(q1) - self.L2*np.sin(q1+q2), -self.L2*np.sin(q1+q2)],
            [self.L1*np.cos(q1) + self.L2*np.cos(q1+q2), self.L2*np.cos(q1+q2)]
        ])
        return J

    def inverse_kinematics_newton(self, target, q_init, max_iter=100, tol=1e-6):
        """
        用牛顿法求逆运动学

        问题：给定目标位置 x_target，找关节角 q 使得 f(q) = x_target

        牛顿法迭代：
            q_{k+1} = q_k + J^{-1}(q_k) * (x_target - f(q_k))

        即：每步根据当前误差，用雅可比逆来估计需要的关节角变化

        参数:
        - target: 目标位置 [x, y]
        - q_init: 初始猜测 [q1, q2]
        - max_iter: 最大迭代次数
        - tol: 收敛容差

        返回:
        - q: 最终关节角
        - history: 迭代历史（用于可视化）

        注意：
        - 逆运动学通常有多个解（或无解）
        - 初始猜测决定收敛到哪个解
        - 奇异点附近 J 不可逆，算法会失败
        """
        q = q_init.copy()
        history = [q.copy()]

        for i in range(max_iter):
            # 计算当前末端位置
            x_current = self.forward_kinematics(q[0], q[1])
            # 计算位置误差
            error = target - x_current

            # 检查是否收敛
            if np.linalg.norm(error) < tol:
                break

            # 计算雅可比矩阵
            J = self.jacobian(q[0], q[1])

            # 检查奇异性（det(J)=0 时无法求逆）
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-6:
                print(f"警告：在迭代{i}步遇到奇异点！det(J) = {det_J:.6f}")
                break

            # 牛顿更新：dq = J^{-1} * error
            dq = np.linalg.solve(J, error)
            q = q + dq
            history.append(q.copy())

        return q, history

# 创建机械臂实例（两个连杆各长1米）
arm = TwoLinkArm(L1=1.0, L2=1.0)

# ===== 正运动学测试 =====
# 这是书中图7.7的例子
# 给定关节角 q1=60°, q2=-30°，计算末端位置
q1 = np.radians(60)  # 第一关节角（60度，转换为弧度）
q2 = np.radians(-30)  # 第二关节角（-30度，相对于第一连杆）
pos = arm.forward_kinematics(q1, q2)
print(f"\n正运动学测试：")
print(f"  输入：q1 = 60°, q2 = -30°")
print(f"  末端位置：x = {pos[0]:.4f}, y = {pos[1]:.4f}")
print(f"  书中结果：x ≈ 1.366, y ≈ 1.366")  # 用于验证

# ===== 逆运动学测试 =====
# 问题：想让末端到达 (1.2, 0.9)，需要什么关节角？
# 方法：用牛顿法迭代求解
target = np.array([1.2, 0.9])  # 目标位置
q_init = np.array([np.radians(30), np.radians(30)])  # 初始猜测
q_solution, ik_history = arm.inverse_kinematics_newton(target, q_init)

print(f"\n逆运动学测试（牛顿法）：")
print(f"  目标位置：x = {target[0]}, y = {target[1]}")
print(f"  初始猜测：q1 = 30°, q2 = 30°")
print(f"  求解结果：q1 = {np.degrees(q_solution[0]):.2f}°, q2 = {np.degrees(q_solution[1]):.2f}°")
# 验证：用正运动学检查求得的解
final_pos = arm.forward_kinematics(q_solution[0], q_solution[1])
print(f"  验证：末端位置 = ({final_pos[0]:.4f}, {final_pos[1]:.4f})")  # 应该接近 (1.2, 0.9)
print(f"  迭代次数：{len(ik_history) - 1}")

# ==================== 可视化 ====================
#
# 创建6个子图展示实验结果
# 这些图对应书中的图7.1, 7.7-7.10, 7.14等
#
fig = plt.figure(figsize=(16, 10))

# ===== 图1：SVD投影结果 =====
# 可视化对比原始矩阵A和投影后的旋转矩阵R
ax1 = fig.add_subplot(231)
# 显示原始矩阵A和投影后的R
im1 = ax1.imshow(np.hstack([A, np.ones((3,1))*np.nan, R]), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks([0.5, 1, 1.5, 3.5, 4, 4.5])
ax1.set_yticks([0, 1, 2])
ax1.set_title('SVD投影到SO(3)\n左:原始A | 右:投影后R')
plt.colorbar(im1, ax=ax1)

# 图2：指数映射演示
ax2 = fig.add_subplot(232, projection='3d')
# 生成旋转轨迹
thetas = np.linspace(0, 2*np.pi, 100)
axis = omega / np.linalg.norm(omega)
points = []
for theta in thetas:
    R_temp = exp_so3(axis * theta)
    # 取第一列作为旋转后的x轴
    points.append(R_temp[:, 0])
points = np.array(points)
ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2)
ax2.scatter([1], [0], [0], color='red', s=100, label='起点')
ax2.set_xlabel('X', labelpad=10)
ax2.set_ylabel('Y', labelpad=10)
ax2.set_zlabel('Z')
ax2.set_title('SO(3)上的旋转轨迹\n(指数映射保持约束)')
ax2.legend()

# 图3：机械臂正运动学
ax3 = fig.add_subplot(233)
def plot_arm(ax, q1, q2, arm, color='blue', label=None):
    """绘制机械臂"""
    x0, y0 = 0, 0
    x1 = arm.L1 * np.cos(q1)
    y1 = arm.L1 * np.sin(q1)
    x2 = x1 + arm.L2 * np.cos(q1 + q2)
    y2 = y1 + arm.L2 * np.sin(q1 + q2)

    ax.plot([x0, x1], [y0, y1], color=color, linewidth=3)
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
    ax.scatter([x0, x1, x2], [y0, y1, y2], color=color, s=100, zorder=5)
    if label:
        ax.scatter([x2], [y2], color=color, s=150, marker='*', label=label, zorder=6)

# 绘制多个姿态
configs = [
    (np.radians(60), np.radians(-30), 'blue', 'q1=60°, q2=-30°'),
    (np.radians(30), np.radians(60), 'green', 'q1=30°, q2=60°'),
    (np.radians(90), np.radians(0), 'red', '奇异: q2=0°'),
]
for q1, q2, color, label in configs:
    plot_arm(ax3, q1, q2, arm, color, label)

# 绘制工作空间边界
theta = np.linspace(0, 2*np.pi, 100)
r_outer = arm.L1 + arm.L2
r_inner = abs(arm.L1 - arm.L2)
ax3.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 'k--', alpha=0.3, label='工作空间')
ax3.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 'k--', alpha=0.3)

ax3.set_xlim(-2.5, 2.5)
ax3.set_ylim(-2.5, 2.5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=8)
ax3.set_title('双连杆平面机械臂\n正运动学')
ax3.set_xlabel('X', labelpad=10)
ax3.set_ylabel('Y', labelpad=10)

# 图4：牛顿法迭代过程
ax4 = fig.add_subplot(234)
ik_history_arr = np.array(ik_history)
# 绘制每次迭代的末端位置
end_positions = []
for q in ik_history:
    pos = arm.forward_kinematics(q[0], q[1])
    end_positions.append(pos)
end_positions = np.array(end_positions)

ax4.plot(end_positions[:, 0], end_positions[:, 1], 'b.-', markersize=10, label='牛顿迭代')
ax4.scatter([end_positions[0, 0]], [end_positions[0, 1]], color='green', s=150, marker='o', label='起点', zorder=5)
ax4.scatter([target[0]], [target[1]], color='red', s=150, marker='*', label='目标', zorder=5)
ax4.set_xlabel('X', labelpad=10)
ax4.set_ylabel('Y', labelpad=10)
ax4.set_title('逆运动学: 牛顿法\n末端轨迹')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# 图5：雅可比矩阵行列式（奇异性分析）
ax5 = fig.add_subplot(235)
q1_range = np.linspace(-np.pi, np.pi, 100)
q2_range = np.linspace(-np.pi, np.pi, 100)
Q1, Q2 = np.meshgrid(q1_range, q2_range)
det_J = np.zeros_like(Q1)

for i in range(len(q1_range)):
    for j in range(len(q2_range)):
        J = arm.jacobian(Q1[i,j], Q2[i,j])
        det_J[i,j] = np.linalg.det(J)

contour = ax5.contourf(np.degrees(Q1), np.degrees(Q2), det_J, levels=50, cmap='RdBu_r')
ax5.contour(np.degrees(Q1), np.degrees(Q2), det_J, levels=[0], colors='black', linewidths=2)
plt.colorbar(contour, ax=ax5, label='det(J)')
ax5.set_xlabel('q1 (度)', labelpad=10)
ax5.set_ylabel('q2 (度)', labelpad=10)
ax5.set_title('雅可比矩阵行列式\n(黑线: 奇异位形)')

# 图6：力传递示例
ax6 = fig.add_subplot(236)
# 书中例子：q1=90°, q2=0°（奇异位形）
q1_singular = np.radians(90)
q2_singular = np.radians(0)

# 非奇异位形对比
q1_normal = np.radians(60)
q2_normal = np.radians(30)

# 绘制两种位形
plot_arm(ax6, q1_singular, q2_singular, arm, 'red', '奇异')
plot_arm(ax6, q1_normal, q2_normal, arm, 'blue', '正常')

# 雅可比分析
J_singular = arm.jacobian(q1_singular, q2_singular)
J_normal = arm.jacobian(q1_normal, q2_normal)

ax6.text(0.05, 0.95, f'奇异 det(J) = {np.linalg.det(J_singular):.4f}',
         transform=ax6.transAxes, fontsize=9, verticalalignment='top', color='red')
ax6.text(0.05, 0.88, f'正常 det(J) = {np.linalg.det(J_normal):.4f}',
         transform=ax6.transAxes, fontsize=9, verticalalignment='top', color='blue')

ax6.set_xlim(-2.5, 2.5)
ax6.set_ylim(-2.5, 2.5)
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)
ax6.legend(loc='upper right', fontsize=8)
ax6.set_title('奇异性分析\n(手臂完全伸直)')
ax6.set_xlabel('X', labelpad=10)
ax6.set_ylabel('Y', labelpad=10)

plt.tight_layout()
plt.savefig('figs/chap07_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap07_fig1.png")

# ==================== 额外验证：力传递 ====================
#
# 这是书中图7.14的关键计算：τ = J^T F
#
# 物理意义：
# - F 是施加在末端的力（笛卡尔空间）
# - τ 是各关节需要的力矩（关节空间）
# - J^T 把末端力映射到关节力矩
#
# 为什么是 J^T 而不是 J？
# - 虚功原理：δW = F·δx = τ·δq
# - 因为 δx = J δq，所以 F^T J δq = τ^T δq
# - 对所有 δq 成立，故 τ = J^T F
#
print("\n" + "="*60)
print("验证：力传递（书中例子）")
print("="*60)
print(f"位形：q1=90°, q2=0°（手臂竖直向上，完全伸直）")
print(f"末端力：F = (10, 0)^T N（水平向右推）")

# 设置末端力
F = np.array([10, 0])  # 水平力 10N

# 计算该位形下的雅可比矩阵
J = arm.jacobian(np.radians(90), np.radians(0))
print(f"\n雅可比矩阵 J:")
print(J)
# 注意：这是一个奇异位形（det(J)=0），但J^T F仍然有意义

# 计算所需的关节力矩：τ = J^T F
# 这就是静力学平衡条件
tau = J.T @ F
print(f"\n关节力矩 τ = J^T @ F:")
print(f"τ = {tau} Nm")
print(f"书中结果：τ = (-20, -10) Nm")  # 验证：与书中一致！
# 负号说明两个关节都需要"反向"力矩来抵抗水平推力

plt.show()
print("\n第7章代码执行完成！")
