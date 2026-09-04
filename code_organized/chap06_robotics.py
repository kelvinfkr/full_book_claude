#!/usr/bin/env python3
"""
第6章：机器人运动学与动力学演示

包含：
1. SVD投影到SO(3)
2. 旋转矩阵的正确更新方式（指数映射）
3. 平面双连杆机械臂的正/逆运动学
4. 雅可比矩阵与奇异性分析
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

print("="*60)
print("第6章：机器人运动学与动力学演示")
print("="*60)

# ==================== 实验1：SVD投影到SO(3) ====================
print("\n" + "="*60)
print("实验1：用SVD把矩阵投影到SO(3)")
print("="*60)

# 一个"有噪声"的矩阵
A = np.array([
    [0.9, -0.5, 0.1],
    [0.4, 0.85, -0.3],
    [-0.1, 0.25, 0.95]
])

print("\nA 不是旋转矩阵：")
print("A^T @ A =\n", np.round(A.T @ A, 3))
print("det(A) =", np.round(np.linalg.det(A), 3))

# 用 SVD 投影到 SO(3)
U, S, Vt = np.linalg.svd(A)
R = U @ Vt

# 确保 det = +1
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = U @ Vt

print("\nR 是旋转矩阵：")
print("R^T @ R =\n", np.round(R.T @ R, 3))
print("det(R) =", np.round(np.linalg.det(R), 3))

# ==================== 实验2：旋转矩阵的正确更新 ====================
print("\n" + "="*60)
print("实验2：旋转矩阵的正确更新方式")
print("="*60)

def skew(w):
    """构造反对称矩阵"""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def exp_so3(omega):
    """指数映射：反对称矩阵 -> 旋转矩阵 (Rodrigues公式)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    w_hat = omega / theta
    K = skew(w_hat)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

R0 = np.eye(3)  # 初始姿态
omega = np.array([0.1, 0.2, 0.3])  # 角速度
dt = 0.1

# 错误的欧拉更新
R_wrong = R0 + dt * R0 @ skew(omega)

print("\n错误的欧拉更新后：")
print("R^T @ R =\n", np.round(R_wrong.T @ R_wrong, 4))
print("det(R) =", np.round(np.linalg.det(R_wrong), 4))
print("结论：不再是旋转矩阵！")

# 正确的更新（指数映射）
delta_R = exp_so3(omega * dt)
R_correct = R0 @ delta_R

print("\n正确的指数映射更新后：")
print("R^T @ R =\n", np.round(R_correct.T @ R_correct, 4))
print("det(R) =", np.round(np.linalg.det(R_correct), 4))
print("结论：仍然是旋转矩阵！")

# ==================== 实验3：平面双连杆机械臂 ====================
print("\n" + "="*60)
print("实验3：平面双连杆机械臂正/逆运动学")
print("="*60)

class TwoLinkArm:
    """平面双连杆机械臂"""

    def __init__(self, L1=1.0, L2=1.0):
        self.L1 = L1
        self.L2 = L2

    def forward_kinematics(self, q1, q2):
        """正运动学：关节角 -> 末端位置"""
        x = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        y = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        return np.array([x, y])

    def jacobian(self, q1, q2):
        """雅可比矩阵"""
        J = np.array([
            [-self.L1*np.sin(q1) - self.L2*np.sin(q1+q2), -self.L2*np.sin(q1+q2)],
            [self.L1*np.cos(q1) + self.L2*np.cos(q1+q2), self.L2*np.cos(q1+q2)]
        ])
        return J

    def inverse_kinematics_newton(self, target, q_init, max_iter=100, tol=1e-6):
        """用牛顿法求逆运动学"""
        q = q_init.copy()
        history = [q.copy()]

        for i in range(max_iter):
            x_current = self.forward_kinematics(q[0], q[1])
            error = target - x_current

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian(q[0], q[1])

            # 检查奇异性
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-6:
                print(f"警告：在迭代{i}步遇到奇异点！det(J) = {det_J:.6f}")
                break

            dq = np.linalg.solve(J, error)
            q = q + dq
            history.append(q.copy())

        return q, history

# 创建机械臂
arm = TwoLinkArm(L1=1.0, L2=1.0)

# 正运动学测试（书中例子）
q1 = np.radians(60)  # 60度
q2 = np.radians(-30)  # -30度
pos = arm.forward_kinematics(q1, q2)
print(f"\n正运动学测试：")
print(f"  输入：q1 = 60°, q2 = -30°")
print(f"  末端位置：x = {pos[0]:.4f}, y = {pos[1]:.4f}")
print(f"  书中结果：x ≈ 1.366, y ≈ 1.366")

# 逆运动学测试
target = np.array([1.2, 0.9])
q_init = np.array([np.radians(30), np.radians(30)])
q_solution, ik_history = arm.inverse_kinematics_newton(target, q_init)

print(f"\n逆运动学测试（牛顿法）：")
print(f"  目标位置：x = {target[0]}, y = {target[1]}")
print(f"  初始猜测：q1 = 30°, q2 = 30°")
print(f"  求解结果：q1 = {np.degrees(q_solution[0]):.2f}°, q2 = {np.degrees(q_solution[1]):.2f}°")
final_pos = arm.forward_kinematics(q_solution[0], q_solution[1])
print(f"  验证：末端位置 = ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
print(f"  迭代次数：{len(ik_history) - 1}")

# ==================== 可视化 ====================
fig = plt.figure(figsize=(16, 10))

# 图1：SVD投影结果
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
plt.savefig('figs/chap06_fig1.png', dpi=150, bbox_inches='tight')
print("\n图像已保存到 figs/chap06_fig1.png")

# 额外验证：力传递（书中例子）
print("\n" + "="*60)
print("验证：力传递（书中例子）")
print("="*60)
print(f"位形：q1=90°, q2=0°（手臂竖直向上）")
print(f"末端力：F = (10, 0)^T N")

F = np.array([10, 0])
J = arm.jacobian(np.radians(90), np.radians(0))
print(f"\n雅可比矩阵 J:")
print(J)

tau = J.T @ F
print(f"\n关节力矩 τ = J^T @ F:")
print(f"τ = {tau} Nm")
print(f"书中结果：τ = (-20, -10) Nm")

plt.show()
print("\n第6章代码执行完成！")
