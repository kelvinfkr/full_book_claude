"""
Turek-Hron FSI2 基准测试可视化
展示弹性板在涡激振动下的周期性摆动
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Turek-Hron FSI2 参数
# =============================================================================

# 几何参数 (单位: m)
L_channel = 2.5      # 通道长度
H_channel = 0.41     # 通道高度
D_cylinder = 0.1     # 圆柱直径
R_cylinder = D_cylinder / 2
cx, cy = 0.2, 0.2    # 圆柱圆心（故意偏心）

# 弹性板参数
flag_length = 0.35   # 板长
flag_thickness = 0.02  # 板厚
flag_start_x = cx + R_cylinder  # 板起始x（固连在圆柱后方）
flag_start_y = cy  # 板起始y（圆柱中心线）

# 物理参数
rho_f = 1000         # 流体密度 kg/m³
mu_f = 0.001         # 动力粘度 Pa·s
rho_s = 10000        # 固体密度 kg/m³
E = 1.4e6            # 杨氏模量 Pa
nu_s = 0.4           # 泊松比
U_mean = 1.0         # 平均入口速度 m/s
Re = rho_f * U_mean * D_cylinder / mu_f  # 雷诺数 = 100

# FSI2 典型结果（来自文献）
y_amplitude = 0.080  # 板端Y方向振幅 ~80mm
freq = 2.0           # 振动频率 ~2 Hz
period = 1.0 / freq


def flag_deformation(s, t, amplitude=y_amplitude, freq=freq, mode=1):
    """
    计算弹性板在给定时刻的变形
    s: 沿板长度的参数 [0, 1]
    t: 时间
    amplitude: 板端振幅
    mode: 振型模态 (1=一阶弯曲)
    """
    # 一阶弯曲模态形状：固定端位移为0，自由端最大
    # 用悬臂梁一阶模态近似: phi(s) ~ 1 - cos(pi*s/2)
    phi = 1 - np.cos(np.pi * s / 2)

    # 时间变化
    omega = 2 * np.pi * freq
    y_displacement = amplitude * phi * np.sin(omega * t)

    return y_displacement


def get_flag_shape(t, n_points=50):
    """获取板在时刻t的形状"""
    s = np.linspace(0, 1, n_points)
    x = flag_start_x + s * flag_length
    y_base = flag_start_y
    y_displacement = flag_deformation(s, t)
    y = y_base + y_displacement
    return x, y


def plot_fsi2_snapshots(save_path):
    """绘制FSI2弹性板振动的多个快照"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # 选择6个时刻展示一个完整周期
    times = [0, period/6, period/3, period/2, 2*period/3, 5*period/6]
    titles = ['t = 0', 't = T/6', 't = T/3', 't = T/2', 't = 2T/3', 't = 5T/6']

    for idx, (ax, t, title) in enumerate(zip(axes.flat, times, titles)):
        # 绘制通道
        ax.plot([0, L_channel], [0, 0], 'k-', linewidth=2)
        ax.plot([0, L_channel], [H_channel, H_channel], 'k-', linewidth=2)
        ax.plot([0, 0], [0, H_channel], 'k-', linewidth=2)

        # 绘制圆柱
        cylinder = Circle((cx, cy), R_cylinder, facecolor='gray', edgecolor='black', linewidth=2)
        ax.add_patch(cylinder)

        # 绘制弹性板（当前变形状态）
        x_flag, y_flag = get_flag_shape(t)
        ax.plot(x_flag, y_flag, 'orange', linewidth=4, solid_capstyle='round')
        ax.plot(x_flag, y_flag + flag_thickness/2, 'darkorange', linewidth=1)
        ax.plot(x_flag, y_flag - flag_thickness/2, 'darkorange', linewidth=1)

        # 绘制初始位置参考线
        ax.plot([flag_start_x, flag_start_x + flag_length],
                [flag_start_y, flag_start_y], 'k--', alpha=0.3, linewidth=1)

        # 入口速度剖面
        y_inlet = np.linspace(0, H_channel, 20)
        u_inlet = 4 * U_mean * y_inlet * (H_channel - y_inlet) / H_channel**2  # 抛物线剖面
        for i in range(0, len(y_inlet), 2):
            ax.arrow(-0.05, y_inlet[i], u_inlet[i]*0.1, 0,
                    head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.5)

        # 涡流示意（简化）
        vortex_phase = 2 * np.pi * freq * t
        for i, vx in enumerate([0.7, 1.0, 1.3, 1.6]):
            vy = cy + 0.05 * np.sin(vortex_phase + i * np.pi)
            sign = 1 if i % 2 == 0 else -1
            vortex = Circle((vx, vy), 0.02, facecolor='none',
                           edgecolor='blue', linewidth=1, alpha=0.5)
            ax.add_patch(vortex)
            # 旋转箭头
            theta = np.linspace(0, 1.5*np.pi, 10)
            ax.plot(vx + 0.02*np.cos(theta), vy + sign*0.02*np.sin(theta),
                   'b-', alpha=0.3, linewidth=0.5)

        ax.set_xlim(-0.15, 0.8)
        ax.set_ylim(-0.02, H_channel + 0.02)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)

        # 标注位移
        y_end = y_flag[-1]
        displacement = y_end - flag_start_y
        ax.annotate(f'Δy = {displacement*1000:.1f} mm',
                   xy=(x_flag[-1], y_end),
                   xytext=(x_flag[-1] + 0.05, y_end + 0.02 * np.sign(displacement)),
                   fontsize=8, color='darkorange',
                   arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.5))

    plt.suptitle('Turek-Hron FSI2: 弹性板周期振动 (T = 0.5s, f ≈ 2Hz)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fsi2_time_history(save_path):
    """绘制板端位移时间历程和相图"""
    fig = plt.figure(figsize=(14, 10))

    # 模拟时间历程
    t = np.linspace(0, 5, 1000)  # 5秒，约10个周期
    omega = 2 * np.pi * freq

    # 板端Y位移（加入启动瞬态）
    envelope = 1 - np.exp(-t / 0.5)  # 启动包络
    y_tip = y_amplitude * envelope * np.sin(omega * t)
    v_tip = y_amplitude * omega * envelope * np.cos(omega * t)  # 速度

    # 左上：Y位移时间历程
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, y_tip * 1000, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=y_amplitude*1000, color='r', linestyle=':', alpha=0.5, label=f'振幅 ±{y_amplitude*1000:.0f} mm')
    ax1.axhline(y=-y_amplitude*1000, color='r', linestyle=':', alpha=0.5)
    ax1.set_xlabel('时间 t (s)', fontsize=11)
    ax1.set_ylabel('板端Y位移 (mm)', fontsize=11)
    ax1.set_title('板端位移时间历程', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # 右上：相图（位移-速度）
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(y_tip[100:] * 1000, v_tip[100:] * 1000, 'b-', linewidth=1, alpha=0.7)
    ax2.plot(y_tip[:100] * 1000, v_tip[:100] * 1000, 'r-', linewidth=1, alpha=0.5, label='瞬态启动')
    ax2.set_xlabel('Y位移 (mm)', fontsize=11)
    ax2.set_ylabel('Y速度 (mm/s)', fontsize=11)
    ax2.set_title('相空间轨迹（极限环）', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 左下：频谱分析
    ax3 = fig.add_subplot(2, 2, 3)
    # FFT
    dt = t[1] - t[0]
    n = len(t)
    freq_axis = np.fft.fftfreq(n, dt)[:n//2]
    y_fft = np.abs(np.fft.fft(y_tip))[:n//2] * 2 / n
    ax3.plot(freq_axis, y_fft * 1000, 'b-', linewidth=1.5)
    ax3.axvline(x=freq, color='r', linestyle='--', alpha=0.7, label=f'主频 f = {freq:.1f} Hz')
    ax3.set_xlabel('频率 (Hz)', fontsize=11)
    ax3.set_ylabel('振幅 (mm)', fontsize=11)
    ax3.set_title('频谱分析', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)

    # 右下：几何示意图
    ax4 = fig.add_subplot(2, 2, 4)

    # 绘制通道
    ax4.plot([0, L_channel], [0, 0], 'k-', linewidth=2)
    ax4.plot([0, L_channel], [H_channel, H_channel], 'k-', linewidth=2)
    ax4.plot([0, 0], [0, H_channel], 'k-', linewidth=2)
    ax4.plot([L_channel, L_channel], [0, H_channel], 'k--', linewidth=1)

    # 圆柱
    cylinder = Circle((cx, cy), R_cylinder, facecolor='gray', edgecolor='black', linewidth=2)
    ax4.add_patch(cylinder)

    # 多个时刻的板位置叠加
    for t_snap in np.linspace(0, period, 12):
        x_flag, y_flag = get_flag_shape(t_snap)
        alpha = 0.3 if t_snap != 0 else 0.8
        ax4.plot(x_flag, y_flag, 'orange', linewidth=2, alpha=alpha)

    # 标注
    ax4.annotate('', xy=(flag_start_x + flag_length, cy + y_amplitude),
                xytext=(flag_start_x + flag_length, cy - y_amplitude),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax4.text(flag_start_x + flag_length + 0.05, cy,
            f'振幅\n±{y_amplitude*1000:.0f}mm', fontsize=10, color='red', va='center')

    # 参数标注
    param_text = (f'FSI2 参数:\n'
                  f'Re = {Re:.0f}\n'
                  f'ρf = {rho_f} kg/m³\n'
                  f'ρs = {rho_s} kg/m³\n'
                  f'E = {E/1e6:.1f} MPa\n'
                  f'f ≈ {freq:.1f} Hz')
    ax4.text(1.5, 0.3, param_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax4.set_xlim(-0.1, L_channel + 0.1)
    ax4.set_ylim(-0.05, H_channel + 0.05)
    ax4.set_aspect('equal')
    ax4.set_xlabel('x (m)', fontsize=11)
    ax4.set_ylabel('y (m)', fontsize=11)
    ax4.set_title('振动包络（一个周期内的位置叠加）', fontsize=12)

    plt.suptitle('Turek-Hron FSI2 基准测试：弹性板自激振荡', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fsi2_mechanism(save_path):
    """绘制FSI机理示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：上摆时刻
    ax1 = axes[0]
    t = period / 4  # 板向上最大位移时刻

    # 通道和圆柱
    ax1.fill([0, 0.8, 0.8, 0], [0, 0, H_channel, H_channel],
            color='lightblue', alpha=0.3)
    ax1.plot([0, 0.8], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, 0.8], [H_channel, H_channel], 'k-', linewidth=2)
    cylinder = Circle((cx, cy), R_cylinder, facecolor='gray', edgecolor='black', linewidth=2)
    ax1.add_patch(cylinder)

    # 弹性板
    x_flag, y_flag = get_flag_shape(t)
    ax1.fill_between(x_flag, y_flag - flag_thickness/2, y_flag + flag_thickness/2,
                    color='orange', alpha=0.8, edgecolor='darkorange', linewidth=2)

    # 流线示意
    for y0 in [0.08, 0.15, 0.25, 0.33]:
        x_stream = np.linspace(-0.05, 0.15, 30)
        y_stream = y0 * np.ones_like(x_stream)
        ax1.plot(x_stream, y_stream, 'b-', alpha=0.5, linewidth=1)
        ax1.arrow(0.1, y0, 0.02, 0, head_width=0.01, head_length=0.005, fc='blue', ec='blue')

    # 圆柱后方的涡
    # 上方涡（逆时针）
    vortex_up = Circle((0.35, 0.28), 0.03, facecolor='none',
                       edgecolor='red', linewidth=2, linestyle='--')
    ax1.add_patch(vortex_up)
    ax1.annotate('', xy=(0.35, 0.31), xytext=(0.38, 0.28),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax1.text(0.35, 0.34, '涡A\n(脱落)', fontsize=8, ha='center', color='red')

    # 下方涡（形成中）
    vortex_down = Circle((0.28, 0.12), 0.025, facecolor='none',
                        edgecolor='blue', linewidth=2)
    ax1.add_patch(vortex_down)
    ax1.annotate('', xy=(0.28, 0.095), xytext=(0.255, 0.12),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5,
                              connectionstyle='arc3,rad=-0.3'))
    ax1.text(0.28, 0.06, '涡B\n(形成中)', fontsize=8, ha='center', color='blue')

    # 压力差示意
    ax1.annotate('低压', xy=(0.4, 0.32), fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.annotate('高压', xy=(0.4, 0.08), fontsize=9, color='blue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 力的方向
    ax1.arrow(0.5, 0.2, 0, 0.06, head_width=0.015, head_length=0.01,
             fc='green', ec='green', linewidth=2)
    ax1.text(0.52, 0.23, 'F升力', fontsize=10, color='green')

    ax1.set_xlim(-0.1, 0.7)
    ax1.set_ylim(-0.02, H_channel + 0.02)
    ax1.set_aspect('equal')
    ax1.set_title('(a) 板向上运动：上方涡脱落，产生向上升力', fontsize=11)
    ax1.set_xlabel('x (m)', fontsize=10)
    ax1.set_ylabel('y (m)', fontsize=10)

    # 右图：下摆时刻
    ax2 = axes[1]
    t = 3 * period / 4  # 板向下最大位移时刻

    # 通道和圆柱
    ax2.fill([0, 0.8, 0.8, 0], [0, 0, H_channel, H_channel],
            color='lightblue', alpha=0.3)
    ax2.plot([0, 0.8], [0, 0], 'k-', linewidth=2)
    ax2.plot([0, 0.8], [H_channel, H_channel], 'k-', linewidth=2)
    cylinder = Circle((cx, cy), R_cylinder, facecolor='gray', edgecolor='black', linewidth=2)
    ax2.add_patch(cylinder)

    # 弹性板
    x_flag, y_flag = get_flag_shape(t)
    ax2.fill_between(x_flag, y_flag - flag_thickness/2, y_flag + flag_thickness/2,
                    color='orange', alpha=0.8, edgecolor='darkorange', linewidth=2)

    # 流线
    for y0 in [0.08, 0.15, 0.25, 0.33]:
        x_stream = np.linspace(-0.05, 0.15, 30)
        y_stream = y0 * np.ones_like(x_stream)
        ax2.plot(x_stream, y_stream, 'b-', alpha=0.5, linewidth=1)
        ax2.arrow(0.1, y0, 0.02, 0, head_width=0.01, head_length=0.005, fc='blue', ec='blue')

    # 涡交换位置
    vortex_down = Circle((0.35, 0.12), 0.03, facecolor='none',
                        edgecolor='blue', linewidth=2, linestyle='--')
    ax2.add_patch(vortex_down)
    ax2.annotate('', xy=(0.35, 0.09), xytext=(0.32, 0.12),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5,
                              connectionstyle='arc3,rad=-0.3'))
    ax2.text(0.35, 0.05, '涡B\n(脱落)', fontsize=8, ha='center', color='blue')

    vortex_up = Circle((0.28, 0.28), 0.025, facecolor='none',
                       edgecolor='red', linewidth=2)
    ax2.add_patch(vortex_up)
    ax2.annotate('', xy=(0.28, 0.305), xytext=(0.305, 0.28),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax2.text(0.28, 0.33, '涡A\n(形成中)', fontsize=8, ha='center', color='red')

    # 压力差
    ax2.annotate('高压', xy=(0.4, 0.32), fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.annotate('低压', xy=(0.4, 0.08), fontsize=9, color='blue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 力的方向
    ax2.arrow(0.5, 0.2, 0, -0.06, head_width=0.015, head_length=0.01,
             fc='green', ec='green', linewidth=2)
    ax2.text(0.52, 0.17, 'F升力', fontsize=10, color='green')

    ax2.set_xlim(-0.1, 0.7)
    ax2.set_ylim(-0.02, H_channel + 0.02)
    ax2.set_aspect('equal')
    ax2.set_title('(b) 板向下运动：下方涡脱落，产生向下升力', fontsize=11)
    ax2.set_xlabel('x (m)', fontsize=10)
    ax2.set_ylabel('y (m)', fontsize=10)

    plt.suptitle('Turek-Hron FSI2：涡激振动机理（卡门涡街交替脱落）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap05"

    print("=" * 60)
    print("Generating Turek-Hron FSI2 visualization figures...")
    print("=" * 60)

    print("\n[1] FSI2 振动快照序列")
    plot_fsi2_snapshots(f"{output_dir}/turek_fsi2_snapshots.pdf")

    print("\n[2] FSI2 时间历程与分析")
    plot_fsi2_time_history(f"{output_dir}/turek_fsi2_time_history.pdf")

    print("\n[3] FSI2 涡激振动机理")
    plot_fsi2_mechanism(f"{output_dir}/turek_fsi2_mechanism.pdf")

    print("\n" + "=" * 60)
    print("All FSI2 figures generated!")
    print("=" * 60)
