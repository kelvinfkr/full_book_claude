#!/usr/bin/env python3
"""
第6章：机器人运动学演示 —— 牛顿法逆运动学
================================================

从仓库根目录运行::

    python3 code/chap06_robotics.py

输出图：figs_chap06/chap06_fig1.{pdf,png}

图的思路（与正文"牛顿迭代的手算演示"完全一致）：
    双连杆机械臂 L1 = L2 = 1，目标 x_d = (1.2, 0.9)，
    初值 q^(0) = (0°, 90°)。
    (a) 每一步迭代后的机械臂姿态：看末端如何逼近目标；
    (b) 末端误差 ||x_d - f(q^(k))|| 随迭代次数（对数轴）：看"二次收敛"
        —— 有效数字每一步大约翻一倍；
    (c) 工作空间上的可操作度 |det J| = L1 L2 |sin q2|：看目标点离奇异位形
        （边界 r = 2 与原点）有多远，解释为什么牛顿法在这里收敛得这么好。

脚本另外在终端打印三个"只需要数、不需要图"的验证：
    1. 用 SVD 把带噪声的矩阵投影回 SO(3)；
    2. 旋转矩阵的更新：欧拉加法会破坏 R^T R = I，指数映射不会；
    3. 力传递 τ = J^T F（正文案例：q = (90°, 0°)、F = (10, 0)）。
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS  # noqa: E402

setup_style()

# ==================== 验证1：SVD 投影到 SO(3) ====================
print("=" * 60)
print("验证1：用 SVD 把矩阵投影到 SO(3)")
print("=" * 60)
A = np.array([[0.9, -0.5, 0.1],
              [0.4, 0.85, -0.3],
              [-0.1, 0.25, 0.95]])
U, S, Vt = np.linalg.svd(A)
R = U @ Vt
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = U @ Vt
print("A^T A =\n", np.round(A.T @ A, 3), "\ndet A =", np.round(np.linalg.det(A), 3))
print("R^T R =\n", np.round(R.T @ R, 3), "\ndet R =", np.round(np.linalg.det(R), 3))


# ==================== 验证2：旋转矩阵的正确更新 ====================
def skew(w):
    """反对称矩阵 [w]_x，满足 [w]_x v = w × v。"""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def exp_so3(omega):
    """指数映射 so(3) -> SO(3)（Rodrigues 公式）。"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


print("\n" + "=" * 60)
print("验证2：旋转矩阵的更新方式")
print("=" * 60)
R0, omega, dt = np.eye(3), np.array([0.1, 0.2, 0.3]), 0.1
R_wrong = R0 + dt * R0 @ skew(omega)
R_right = R0 @ exp_so3(omega * dt)
print("欧拉加法： ||R^T R - I|| =", f"{np.linalg.norm(R_wrong.T @ R_wrong - np.eye(3)):.2e}")
print("指数映射： ||R^T R - I|| =", f"{np.linalg.norm(R_right.T @ R_right - np.eye(3)):.2e}")


# ==================== 双连杆机械臂 ====================
class TwoLinkArm:
    """平面双连杆机械臂：q1 为大臂绝对角，q2 为小臂相对大臂的角。"""

    def __init__(self, L1=1.0, L2=1.0):
        self.L1, self.L2 = L1, L2

    def joints(self, q):
        """返回基座、肘关节、末端三点坐标（3×2）。"""
        q1, q2 = q
        p1 = np.array([self.L1 * np.cos(q1), self.L1 * np.sin(q1)])
        p2 = p1 + np.array([self.L2 * np.cos(q1 + q2), self.L2 * np.sin(q1 + q2)])
        return np.vstack([[0.0, 0.0], p1, p2])

    def forward_kinematics(self, q):
        return self.joints(q)[2]

    def jacobian(self, q):
        q1, q2 = q
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        return np.array([[-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
                         [self.L1 * c1 + self.L2 * c12, self.L2 * c12]])

    def inverse_kinematics_newton(self, target, q_init, max_iter=20, tol=1e-15):
        """牛顿法：q_{k+1} = q_k + J(q_k)^{-1} (x_d - f(q_k))。

        返回迭代序列 q^(0..K) 与每一步的末端误差 ||x_d - f(q^(k))||。
        """
        q = np.array(q_init, dtype=float)
        history, errors = [q.copy()], []
        for _ in range(max_iter):
            e = target - self.forward_kinematics(q)
            errors.append(np.linalg.norm(e))
            if errors[-1] < tol:
                break
            J = self.jacobian(q)
            if abs(np.linalg.det(J)) < 1e-9:
                print("警告：遇到奇异位形，det J ≈ 0，停止迭代")
                break
            q = q + np.linalg.solve(J, e)
            history.append(q.copy())
        else:
            errors.append(np.linalg.norm(target - self.forward_kinematics(q)))
        return np.array(history), np.array(errors)


arm = TwoLinkArm(L1=1.0, L2=1.0)

# ---- 正文算例：目标 (1.2, 0.9)，初值 (0°, 90°) ----
target = np.array([1.2, 0.9])
q0 = np.radians([0.0, 90.0])
Q_hist, err = arm.inverse_kinematics_newton(target, q0)

print("\n" + "=" * 60)
print("牛顿法逆运动学（正文算例）：L1 = L2 = 1，x_d = (1.2, 0.9)，q(0) = (0°, 90°)")
print("=" * 60)
print(f"{'k':>2} | {'q1 (deg)':>9} {'q2 (deg)':>9} | {'末端 x':>7} {'末端 y':>7} | 误差 ||e||")
for k, q in enumerate(Q_hist):
    x = arm.forward_kinematics(q)
    print(f"{k:>2} | {np.degrees(q[0]):>9.4f} {np.degrees(q[1]):>9.4f} | "
          f"{x[0]:>7.4f} {x[1]:>7.4f} | {err[k]:.3e}")
q_star = Q_hist[-1]
print(f"\n收敛解 q* = ({np.degrees(q_star[0]):.2f}°, {np.degrees(q_star[1]):.2f}°)；"
      f"余弦定理检验 cos q2 = {(1.2**2 + 0.9**2 - 2) / 2:.3f}，"
      f"cos(q2*) = {np.cos(q_star[1]):.3f}")

# ---- 正文 tryit：初值 (30°, 30°) 的第一步 ----
Q_bad, err_bad = arm.inverse_kinematics_newton(target, np.radians([30.0, 30.0]))
dq_bad = np.degrees(Q_bad[1] - Q_bad[0])
print(f"换初值 (30°, 30°)：第一步 Δq = ({dq_bad[0]:.1f}°, {dq_bad[1]:.1f}°)，"
      f"误差 {err_bad[0]:.3f} -> {err_bad[1]:.3f}（先变大再收敛，共 {len(err_bad) - 1} 步）")

# ==================== 验证3：力传递 τ = J^T F ====================
print("\n" + "=" * 60)
print("验证3：力传递 τ = J^T F（q = (90°, 0°)，F = (10, 0) N）")
print("=" * 60)
J_up = arm.jacobian(np.radians([90.0, 0.0]))
print("J =\n", np.round(J_up, 3))
print("τ = J^T F =", np.round(J_up.T @ np.array([10.0, 0.0]), 3), "N·m   （det J =",
      f"{np.linalg.det(J_up):.1e}，奇异位形）")

# ==================== 绘图 ====================
fig = plt.figure(figsize=(6.3, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[0.72, 1.28],
                      wspace=0.30, hspace=0.55, left=0.08, right=0.99, top=0.96, bottom=0.09)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# ---------- (a) 迭代过程中的机械臂姿态 ----------
ax = ax_a
P0, P1, Ps = arm.joints(Q_hist[0]), arm.joints(Q_hist[1]), arm.joints(q_star)
ax.plot(P0[:, 0], P0[:, 1], '-o', color=COLORS['blue'], alpha=0.35, lw=2.4, markersize=4,
        zorder=3, label=r'$k=0$：$q^{(0)}=(0^\circ,\,90^\circ)$')
ax.plot(P1[:, 0], P1[:, 1], '-o', color=COLORS['blue'], lw=2.4, markersize=4, zorder=4,
        label=r'$k=1$：$q^{(1)}=(-5.7^\circ,\,84.3^\circ)$')
ax.plot(Ps[:, 0], Ps[:, 1], '--', color=COLORS['black'], lw=1.3, zorder=5,
        label=r'收敛解 $q^\ast=(-4.5^\circ,\,82.8^\circ)$')
ends = np.array([arm.forward_kinematics(q) for q in Q_hist[:3]])
ax.plot(ends[:, 0], ends[:, 1], ':', color=COLORS['gray'], lw=1.0, zorder=2)
ax.plot(target[0], target[1], '*', color=COLORS['red'], markersize=12, zorder=6,
        markeredgecolor='white', markeredgewidth=0.5, label=r'目标 $x_d=(1.2,\,0.9)$')
ax.plot(0, 0, 's', color=COLORS['black'], markersize=6, zorder=7)
ax.text(0.03, 0.05, '基座', fontsize=8, ha='left', va='bottom')
ax.text(1.05, 0.60, r'$k=0$', fontsize=8.5, color=COLORS['blue'], alpha=0.7, ha='left')
ax.text(1.14, 0.30, r'$k=1$', fontsize=8.5, color=COLORS['blue'], ha='left')
ax.annotate(r'$x^{(0)}=(1,1)$', xy=ends[0], xytext=(0.62, 1.16), fontsize=8,
            color=COLORS['blue'], alpha=0.8,
            arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.6))
ax.set_xlim(-0.22, 1.66)
ax.set_ylim(-0.72, 1.32)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend(loc='lower left', fontsize=7.3, handlelength=1.8, borderaxespad=0.3)
panel_label(ax, '(a)')

# 目标附近放大：k=1 与 k=2 的末端与目标的距离
axins = ax.inset_axes([0.13, 0.48, 0.36, 0.30])
axins.plot(ends[1, 0], ends[1, 1], 'o', color=COLORS['blue'], markersize=5)
axins.plot(ends[2, 0], ends[2, 1], 'o', color=COLORS['black'], markersize=5)
axins.plot(target[0], target[1], '*', color=COLORS['red'], markersize=10,
           markeredgecolor='white', markeredgewidth=0.5)
axins.annotate(r'$x^{(1)}$', xy=ends[1], xytext=(6, -2), textcoords='offset points',
               fontsize=7.5, color=COLORS['blue'], ha='left', va='center')
axins.annotate(r'$x^{(2)}\approx x_d$', xy=ends[2], xytext=(4, 5), textcoords='offset points',
               fontsize=7.5, color=COLORS['black'], ha='left')
axins.set_xlim(1.18, 1.235)
axins.set_ylim(0.865, 0.92)
axins.set_xticks([1.19, 1.22])
axins.set_yticks([0.87, 0.90])
axins.tick_params(labelsize=6.5, length=2)
axins.grid(True, alpha=0.25)
axins.set_title('目标附近放大', fontsize=7.5, pad=2)
for sp in axins.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.6)

# ---------- (b) 末端误差的二次收敛 ----------
ax = ax_b
ks = np.arange(len(err))
err_plot = np.maximum(err, 1e-17)  # 最后一步误差可能恰好为 0，对数轴需要正数
ax.semilogy(ks, err_plot, '-o', color=COLORS['blue'], lw=1.6, markersize=5, zorder=3)
ax.axhline(2.2e-16, color=COLORS['gray'], ls='--', lw=1.0)
ax.text(0.0, 6e-16, '双精度机器精度', fontsize=7.5, color=COLORS['gray'], ha='left', va='bottom')
for k, e in zip(ks, err):
    if e > 1e-15:
        mant, expo = f'{e:.1e}'.split('e')
        ax.annotate(rf'${mant}\times10^{{{int(expo)}}}$', xy=(k, e), xytext=(7, 2),
                    textcoords='offset points', fontsize=7.5, color=COLORS['blue'])
ax.set_xlabel(r'迭代次数 $k$', labelpad=1)
ax.set_ylabel(r'末端误差 $\|e_k\|$')
ax.set_xticks(ks)
ax.set_ylim(5e-18, 5)
ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
panel_label(ax, '(b)', x=-0.16, y=1.02)

# ---------- (c) 工作空间上的可操作度 |det J| ----------
ax = ax_c
r_e = np.linspace(0, arm.L1 + arm.L2, 201)
phi_e = np.linspace(0, 2 * np.pi, 362)
r_c = 0.5 * (r_e[1:] + r_e[:-1])
# 给定末端半径 r，由余弦定理得 cos q2，于是 |det J| = L1 L2 |sin q2|
cos_q2 = np.clip((r_c**2 - arm.L1**2 - arm.L2**2) / (2 * arm.L1 * arm.L2), -1, 1)
manip = np.tile(arm.L1 * arm.L2 * np.sqrt(1 - cos_q2**2), (len(phi_e) - 1, 1))
Rg, Pg = np.meshgrid(r_e, phi_e)
pc = ax.pcolormesh(Rg * np.cos(Pg), Rg * np.sin(Pg), manip, cmap='Blues',
                   shading='flat', vmin=0, vmax=1, rasterized=True)
cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.03)
cb.set_label(r'$|\det J| = L_1 L_2 |\sin q_2|$', fontsize=8)
cb.ax.tick_params(labelsize=7.5)
tt = np.linspace(0, 2 * np.pi, 200)
ax.plot(2 * np.cos(tt), 2 * np.sin(tt), '--', color=COLORS['red'], lw=1.2)
ax.plot(np.sqrt(2) * np.cos(tt), np.sqrt(2) * np.sin(tt), ':', color=COLORS['black'], lw=1.0)
ax.plot(Ps[:, 0], Ps[:, 1], '-o', color=COLORS['blue'], lw=1.8, markersize=3.5, zorder=4)
ax.plot(target[0], target[1], '*', color=COLORS['red'], markersize=10, zorder=6,
        markeredgecolor='white', markeredgewidth=0.5)
ax.text(0, 2.12, r'边界 $r=2$：伸直，$\det J=0$', fontsize=7.5, color=COLORS['red'],
        ha='center', va='bottom')
ax.annotate(r'$r=\sqrt{2}$：最灵活', xy=(-1.0, -1.0), xytext=(-2.45, -2.3),
            fontsize=7.5, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.6))
ax.annotate(r'目标：$r=1.5$', xy=(target[0], target[1]), xytext=(2.45, -2.3),
            fontsize=7.5, color=COLORS['red'], ha='right', va='top',
            arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.6))
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.85, 2.55)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
ax.set_xlabel(r'$x$', labelpad=-4)
ax.set_ylabel(r'$y$')
panel_label(ax, '(c)', x=-0.16, y=1.0)

save_figure(fig, 'figs_chap06/chap06_fig1')
print("\n第6章代码执行完成！")
