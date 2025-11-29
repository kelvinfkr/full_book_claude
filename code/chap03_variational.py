"""
第3章：变分法 - 可视化演示

包含：
1. 最速降线问题：摆线 vs 直线 vs 抛物线
2. 悬链线问题：悬链线 vs 抛物线
3. 测地线问题：球面上的最短路径
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()

def create_brachistochrone_figure():
    """
    最速降线问题：从A点(0,0)到B点(x_B, y_B)

    比较三种曲线的下滑时间：
    1. 直线
    2. 抛物线
    3. 摆线（最速降线）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 目标点
    x_B, y_B = 4.0, -3.0
    g = 9.8

    # === 子图1: 三种曲线对比 ===
    ax1 = axes[0]

    # 1. 直线
    x_line = np.linspace(0, x_B, 100)
    y_line = (y_B / x_B) * x_line

    # 2. 抛物线 (过原点和B点)
    # y = a*x^2，代入B点得 a = y_B / x_B^2
    a_para = y_B / x_B**2
    x_para = np.linspace(0, x_B, 100)
    y_para = a_para * x_para**2

    # 3. 摆线 (Cycloid)
    # 参数方程: x = R(θ - sinθ), y = -R(1 - cosθ)
    # 需要找到R和θ_B使得摆线过点(x_B, y_B)
    def cycloid_endpoint(R, theta_B):
        x = R * (theta_B - np.sin(theta_B))
        y = -R * (1 - np.cos(theta_B))
        return x - x_B, y - y_B

    # 数值求解R和θ_B
    def objective(params):
        R, theta_B = params
        dx, dy = cycloid_endpoint(R, theta_B)
        return dx**2 + dy**2

    from scipy.optimize import minimize
    result = minimize(objective, [2.0, 2.5], method='Nelder-Mead')
    R_opt, theta_B_opt = result.x

    theta_cyc = np.linspace(0, theta_B_opt, 100)
    x_cyc = R_opt * (theta_cyc - np.sin(theta_cyc))
    y_cyc = -R_opt * (1 - np.cos(theta_cyc))

    # 绘制三条曲线
    ax1.plot(x_line, y_line, 'b--', linewidth=2, label='直线')
    ax1.plot(x_para, y_para, 'g-.', linewidth=2, label='抛物线')
    ax1.plot(x_cyc, y_cyc, 'r-', linewidth=3, label='摆线(最速降线)')

    # 标记起点和终点
    ax1.plot(0, 0, 'ko', markersize=10)
    ax1.plot(x_B, y_B, 'ko', markersize=10)
    ax1.annotate('A (0, 0)', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=12)
    ax1.annotate(f'B ({x_B}, {y_B})', (x_B, y_B), textcoords="offset points", xytext=(10, -15), fontsize=12)

    ax1.set_xlabel('x (米)', fontsize=12, labelpad=10)
    ax1.set_ylabel('y (米)', fontsize=12, labelpad=10)
    ax1.set_title('最速降线：路径对比', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    # 注意：y值已经是负数（y_B=-3），不需要invert_yaxis()
    # 标准坐标系中，负y值自然在下方，正确表示下落

    # === 子图2: 下滑时间计算 ===
    ax2 = axes[1]

    def compute_descent_time(x_arr, y_arr):
        """计算沿曲线下滑的时间（使用能量守恒）"""
        # v = sqrt(2g|y|)，时间 = 积分 ds/v
        total_time = 0
        for i in range(len(x_arr) - 1):
            dx = x_arr[i+1] - x_arr[i]
            dy = y_arr[i+1] - y_arr[i]
            ds = np.sqrt(dx**2 + dy**2)
            y_mid = (abs(y_arr[i]) + abs(y_arr[i+1])) / 2
            if y_mid > 1e-10:
                v = np.sqrt(2 * g * y_mid)
                total_time += ds / v
        return total_time

    t_line = compute_descent_time(x_line, y_line)
    t_para = compute_descent_time(x_para, y_para)
    t_cyc = compute_descent_time(x_cyc, y_cyc)

    methods = ['直线', '抛物线', '摆线']
    times = [t_line, t_para, t_cyc]
    colors = ['blue', 'green', 'red']

    bars = ax2.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('下滑时间 (秒)', fontsize=12, labelpad=10)
    ax2.set_title('下滑时间对比', fontsize=14)

    # 在柱子上标注数值
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylim(0, max(times) * 1.2)

    # === 子图3: 摆线的几何生成 ===
    ax3 = axes[2]

    # 滚动的圆
    R_demo = 1.0
    theta_vals = np.linspace(0, 2*np.pi, 100)

    # 画出摆线轨迹
    x_demo = R_demo * (theta_vals - np.sin(theta_vals))
    y_demo = R_demo * (1 - np.cos(theta_vals))
    ax3.plot(x_demo, y_demo, 'r-', linewidth=2, label='摆线轨迹')

    # 画几个位置的圆
    for theta in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
        cx = R_demo * theta  # 圆心x坐标
        cy = R_demo          # 圆心y坐标
        circle = plt.Circle((cx, cy), R_demo, fill=False, color='gray', linestyle='--')
        ax3.add_patch(circle)

        # 标记圆上的点
        px = cx - R_demo * np.sin(theta)
        py = cy - R_demo * np.cos(theta)
        ax3.plot(px, py, 'ro', markersize=6)

    # 画基线
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax3.set_xlabel('x', fontsize=12, labelpad=10)
    ax3.set_ylabel('y', fontsize=12, labelpad=10)
    ax3.set_title('摆线的几何生成：滚动的圆', fontsize=14)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=10)
    ax3.set_xlim(-0.5, 7)
    ax3.set_ylim(-0.5, 2.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap03_fig1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap03_fig1.png")
    print(f"下滑时间对比: 直线={t_line:.3f}s, 抛物线={t_para:.3f}s, 摆线={t_cyc:.3f}s")
    print(f"摆线比直线快 {(t_line - t_cyc)/t_line * 100:.1f}%")


def create_catenary_figure():
    """
    悬链线问题：悬挂的链条形状

    比较：
    1. 悬链线 y = a * cosh(x/a)
    2. 抛物线近似
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === 子图1: 悬链线 vs 抛物线 ===
    ax1 = axes[0]

    # 悬链线参数
    a = 1.0  # 悬链线参数
    x_range = np.linspace(-2, 2, 200)

    # 悬链线
    y_catenary = a * np.cosh(x_range / a)

    # 抛物线近似（在x=0处展开cosh到2阶）
    # cosh(x/a) ≈ 1 + x²/(2a²)
    y_parabola = a * (1 + x_range**2 / (2 * a**2))

    ax1.plot(x_range, -y_catenary + y_catenary[len(y_catenary)//2], 'b-',
             linewidth=2.5, label=f'悬链线: y = a·cosh(x/a), a={a}')
    ax1.plot(x_range, -y_parabola + y_parabola[len(y_parabola)//2], 'r--',
             linewidth=2, label='抛物线近似')

    # 标记悬挂点
    ax1.plot([-2, 2], [-y_catenary[0] + y_catenary[len(y_catenary)//2]] * 2,
             'ko', markersize=10)

    ax1.set_xlabel('x', fontsize=12, labelpad=10)
    ax1.set_ylabel('y', fontsize=12, labelpad=10)
    ax1.set_title('悬链线 vs 抛物线 (a=1)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # === 子图2: 不同a值的悬链线 ===
    ax2 = axes[1]

    a_values = [0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(a_values)))

    for a_val, color in zip(a_values, colors):
        y = a_val * np.cosh(x_range / a_val)
        y_shifted = y - y[len(y)//2]  # 让最低点在y=0
        ax2.plot(x_range, y_shifted, color=color, linewidth=2, label=f'a = {a_val}')

    ax2.set_xlabel('x', fontsize=12, labelpad=10)
    ax2.set_ylabel('y (平移后)', fontsize=12, labelpad=10)
    ax2.set_title('不同参数a的悬链线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === 子图3: 误差分析 ===
    ax3 = axes[2]

    x_err = np.linspace(0, 2, 100)

    for a_val in [0.5, 1.0, 2.0]:
        exact = np.cosh(x_err / a_val)
        approx = 1 + x_err**2 / (2 * a_val**2)
        relative_error = np.abs(exact - approx) / exact * 100
        ax3.semilogy(x_err, relative_error, linewidth=2, label=f'a = {a_val}')

    ax3.set_xlabel('x', fontsize=12, labelpad=10)
    ax3.set_ylabel('相对误差 (%)', fontsize=12, labelpad=10)
    ax3.set_title('抛物线近似的误差', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-4, 100)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap03_fig2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap03_fig2.png")


def create_geodesic_figure():
    """
    测地线：球面上的最短路径（大圆弧）
    """
    fig = plt.figure(figsize=(12, 5))

    # === 子图1: 球面上的大圆 ===
    ax1 = fig.add_subplot(121, projection='3d')

    # 画球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.3, color='lightblue')

    # 画赤道（一个大圆）
    theta_eq = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta_eq), np.sin(theta_eq), np.zeros_like(theta_eq),
             'b-', linewidth=2, label='赤道(测地线)')

    # 画一条经线（也是大圆）
    ax1.plot(np.cos(theta_eq), np.zeros_like(theta_eq), np.sin(theta_eq),
             'r-', linewidth=2, label='经线(测地线)')

    # 标记两点和它们之间的最短路径
    phi1, theta1 = np.pi/4, np.pi/6
    phi2, theta2 = np.pi/4, 2*np.pi/3

    p1 = [np.sin(phi1)*np.cos(theta1), np.sin(phi1)*np.sin(theta1), np.cos(phi1)]
    p2 = [np.sin(phi2)*np.cos(theta2), np.sin(phi2)*np.sin(theta2), np.cos(phi2)]

    ax1.scatter(*p1, color='green', s=100, zorder=5)
    ax1.scatter(*p2, color='green', s=100, zorder=5)

    ax1.set_xlabel('X', labelpad=10)
    ax1.set_ylabel('Y', labelpad=10)
    ax1.set_zlabel('Z')
    ax1.set_title('球面上的测地线', fontsize=14)
    ax1.legend(fontsize=9)

    # === 子图2: 局部测地线示意 ===
    ax2 = fig.add_subplot(122)

    # 画一个曲面的局部
    x_surf = np.linspace(-2, 2, 50)
    y_surf = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x_surf, y_surf)
    Z = 0.2 * (X**2 + Y**2)  # 抛物面

    ax2.contour(X, Y, Z, levels=15, cmap='Blues', alpha=0.5)

    # 画测地线（直线在投影中不是直线）
    t = np.linspace(0, 1, 50)
    # 直线路径
    x_straight = -1.5 + 3 * t
    y_straight = -1 + 2 * t
    ax2.plot(x_straight, y_straight, 'r--', linewidth=2, label='投影中的直线')

    # 测地线（略微弯曲）
    x_geo = -1.5 + 3 * t + 0.3 * np.sin(np.pi * t)
    y_geo = -1 + 2 * t + 0.2 * np.sin(np.pi * t)
    ax2.plot(x_geo, y_geo, 'g-', linewidth=2, label='曲面上的测地线')

    ax2.plot([-1.5, 1.5], [-1, 1], 'ko', markersize=8)
    ax2.annotate('A', (-1.5, -1), textcoords="offset points", xytext=(-10, -10), fontsize=12)
    ax2.annotate('B', (1.5, 1), textcoords="offset points", xytext=(10, 10), fontsize=12)

    ax2.set_xlabel('x', fontsize=12, labelpad=10)
    ax2.set_ylabel('y', fontsize=12, labelpad=10)
    ax2.set_title('弯曲曲面上的测地线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap03_fig3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap03_fig3.png")


if __name__ == "__main__":
    print("=== 第3章：变分法可视化 ===\n")

    print("1. 生成最速降线图...")
    create_brachistochrone_figure()

    print("\n2. 生成悬链线图...")
    create_catenary_figure()

    print("\n3. 生成测地线图...")
    create_geodesic_figure()

    print("\n所有图像生成完成!")
