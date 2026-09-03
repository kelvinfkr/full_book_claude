"""
Turek-Hron FSI2 基准测试可视化
展示弹性板在涡激振动下的周期性摆动
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

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
mu_f = 1.0           # 动力粘度 Pa·s (FSI2用的是这个值使得Re=100)
rho_s = 10000        # 固体密度 kg/m³
E = 1.4e6            # 杨氏模量 Pa
nu_s = 0.4           # 泊松比
U_mean = 1.0         # 平均入口速度 m/s
Re = 100             # 雷诺数 (FSI2标准值)

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
    phi = 1 - np.cos(np.pi * s / 2)
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


def generate_velocity_field(t, nx=100, ny=50):
    """
    生成卡门涡街速度场的模拟数据
    涡随时间向下游对流移动，体现真实的涡街演化
    """
    x = np.linspace(0, 1.2, nx)
    y = np.linspace(0, H_channel, ny)
    X, Y = np.meshgrid(x, y)

    # 基础速度场 - 抛物线入口剖面
    u_base = 4 * U_mean * Y * (H_channel - Y) / H_channel**2
    v_base = np.zeros_like(X)

    # 初始化速度场
    U = u_base.copy()
    V = v_base.copy()

    # 卡门涡街参数
    convection_speed = 0.8 * U_mean  # 涡的对流速度（略小于主流速度）
    vortex_spacing = 0.12  # 涡间距（沿x方向）
    lateral_spacing = 0.07  # 涡上下偏移
    shedding_period = vortex_spacing / convection_speed  # 涡脱落周期

    # 生成对流移动的涡
    # 涡从圆柱后方(x≈0.3)产生，向下游移动
    n_vortices = 12
    for i in range(n_vortices):
        # 涡的初始位置（脱落时刻）
        shed_time = i * shedding_period / 2  # 每半周期脱落一个涡

        # 当前时刻涡的x位置 = 初始位置 + 对流距离
        # 涡从圆柱后方开始，随时间向右移动
        vortex_x_initial = 0.32  # 脱落起始位置
        vortex_x = vortex_x_initial + convection_speed * (t - shed_time)

        # 只绘制在可见范围内的涡
        if vortex_x < 0.28 or vortex_x > 1.15:
            continue

        # 交替上下 - 形成经典卡门涡街模式
        if i % 2 == 0:
            vortex_y = cy + lateral_spacing
            circulation = 0.018  # 正环量（逆时针）
        else:
            vortex_y = cy - lateral_spacing
            circulation = -0.018  # 负环量（顺时针）

        # 涡强度随距离衰减（模拟粘性耗散）
        distance_from_cylinder = vortex_x - cx
        decay = np.exp(-distance_from_cylinder / 1.5)
        circulation *= decay

        # Rankine涡模型
        core_radius = 0.035
        dx = X - vortex_x
        dy = Y - vortex_y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2 + 1e-10)

        # 核心内刚体旋转，核心外势涡
        factor = np.where(r < core_radius,
                          r / (core_radius**2),
                          1 / (r + 0.005))

        # 涡诱导速度（垂直于径向）
        U += circulation * (-dy) * factor / (2 * np.pi)
        V += circulation * dx * factor / (2 * np.pi)

    # 圆柱绕流修正
    dx_cyl = X - cx
    dy_cyl = Y - cy
    r_cyl = np.sqrt(dx_cyl**2 + dy_cyl**2)

    # 尾流区减速
    wake_factor = np.where((r_cyl > R_cylinder) & (dx_cyl > 0),
                            1 - 0.6 * np.exp(-(r_cyl - R_cylinder) / 0.08) * np.exp(-(dy_cyl**2) / 0.01),
                            np.ones_like(X))
    U *= wake_factor

    # 圆柱内部速度置零
    cylinder_mask = r_cyl < R_cylinder * 1.1
    U[cylinder_mask] = np.nan
    V[cylinder_mask] = np.nan

    # 弹性板遮罩
    x_flag, y_flag = get_flag_shape(t, n_points=100)
    for j in range(ny):
        for k in range(nx):
            if X[j, k] >= flag_start_x and X[j, k] <= flag_start_x + flag_length:
                idx = int((X[j, k] - flag_start_x) / flag_length * 99)
                idx = min(idx, 99)
                if abs(Y[j, k] - y_flag[idx]) < flag_thickness * 1.5:
                    U[j, k] = np.nan
                    V[j, k] = np.nan

    # 速度大小
    speed = np.sqrt(U**2 + V**2)

    return X, Y, U, V, speed


def generate_vorticity_field(t, nx=150, ny=80):
    """
    生成卡门涡街涡量场的模拟数据（备用）
    """
    x = np.linspace(0, 1.2, nx)
    y = np.linspace(0, H_channel, ny)
    X, Y = np.meshgrid(x, y)

    # 涡量场初始化
    vorticity = np.zeros_like(X)

    # 卡门涡街参数
    St = 0.2  # Strouhal数
    vortex_freq = St * U_mean / D_cylinder  # 涡脱落频率
    wavelength = U_mean / vortex_freq  # 涡街波长

    omega_t = 2 * np.pi * freq * t

    # 生成交替的涡
    for i in range(8):
        # 涡的位置随时间向下游移动
        vortex_x = 0.35 + i * wavelength * 0.4 - (t % period) * U_mean * 0.5

        if vortex_x < 0.25 or vortex_x > 1.15:
            continue

        # 交替上下
        if i % 2 == 0:
            vortex_y = cy + 0.06
            sign = 1  # 正涡量（逆时针）
        else:
            vortex_y = cy - 0.06
            sign = -1  # 负涡量（顺时针）

        # 高斯涡核
        r2 = (X - vortex_x)**2 + (Y - vortex_y)**2
        sigma = 0.025
        vortex_strength = sign * 2.0 * np.exp(-r2 / (2 * sigma**2))
        vorticity += vortex_strength

    # 圆柱遮罩
    cylinder_mask = (X - cx)**2 + (Y - cy)**2 < R_cylinder**2
    vorticity[cylinder_mask] = 0

    # 弹性板遮罩
    x_flag, y_flag = get_flag_shape(t, n_points=100)
    for j in range(ny):
        for i in range(nx):
            if X[j, i] >= flag_start_x and X[j, i] <= flag_start_x + flag_length:
                # 插值找到板在这个x位置的y值
                idx = int((X[j, i] - flag_start_x) / flag_length * 99)
                idx = min(idx, 99)
                if abs(Y[j, i] - y_flag[idx]) < flag_thickness:
                    vorticity[j, i] = 0

    return X, Y, vorticity


def plot_fsi2_snapshots(save_path):
    """绘制FSI2弹性板振动的多个快照"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

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
        u_inlet = 4 * U_mean * y_inlet * (H_channel - y_inlet) / H_channel**2
        for i in range(0, len(y_inlet), 2):
            ax.arrow(-0.05, y_inlet[i], u_inlet[i]*0.1, 0,
                    head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.5)

        # 涡流示意
        vortex_phase = 2 * np.pi * freq * t
        for i, vx in enumerate([0.7, 1.0, 1.3, 1.6]):
            vy = cy + 0.05 * np.sin(vortex_phase + i * np.pi)
            sign = 1 if i % 2 == 0 else -1
            vortex = Circle((vx, vy), 0.02, facecolor='none',
                           edgecolor='blue', linewidth=1, alpha=0.5)
            ax.add_patch(vortex)
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
        if abs(displacement) > 0.001:
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
    t = np.linspace(0, 5, 1000)
    omega = 2 * np.pi * freq

    # 板端Y位移（加入启动瞬态）
    envelope = 1 - np.exp(-t / 0.5)
    y_tip = y_amplitude * envelope * np.sin(omega * t)
    v_tip = y_amplitude * omega * envelope * np.cos(omega * t)

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

    # 右下：几何示意图 - 修复布局问题
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

    # 振幅标注 - 放在板端右侧
    ax4.annotate('', xy=(flag_start_x + flag_length + 0.02, cy + y_amplitude),
                xytext=(flag_start_x + flag_length + 0.02, cy - y_amplitude),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax4.text(flag_start_x + flag_length + 0.08, cy,
            f'振幅\n±{y_amplitude*1000:.0f}mm', fontsize=10, color='red', va='center')

    # 参数标注 - 放在右上角，避免与图形重叠
    param_text = (f'FSI2 参数:\n'
                  f'Re = {Re:.0f}\n'
                  f'ρf = {rho_f} kg/m³\n'
                  f'ρs = {rho_s} kg/m³\n'
                  f'E = {E/1e6:.1f} MPa\n'
                  f'f ≈ {freq:.1f} Hz')
    ax4.text(0.98, 0.98, param_text, fontsize=9, transform=ax4.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

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
    """绘制FSI机理示意图 - 使用速度场和流线可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (ax, t_phase, subtitle) in enumerate(zip(
        axes,
        [period/4, 3*period/4],
        ['(a) 板向上运动：上方涡脱落', '(b) 板向下运动：下方涡脱落']
    )):
        # 生成速度场
        X, Y, U, V, speed = generate_velocity_field(t_phase, nx=80, ny=40)

        # 绘制速度大小云图
        levels = np.linspace(0, 1.8, 19)
        cf = ax.contourf(X, Y, speed, levels=levels, cmap='coolwarm', extend='max', alpha=0.9)

        # 绘制流线 - 更自然地展示涡街
        seed_points = []
        for y0 in np.linspace(0.03, H_channel-0.03, 12):
            seed_points.append([0.01, y0])
        seed_points = np.array(seed_points)

        # 使用streamplot绘制流线
        strm = ax.streamplot(X, Y, U, V, color='white', density=1.5, linewidth=0.8,
                             arrowsize=0.8, arrowstyle='->', broken_streamlines=True)

        # 通道边界
        ax.plot([0, 1.2], [0, 0], 'k-', linewidth=3)
        ax.plot([0, 1.2], [H_channel, H_channel], 'k-', linewidth=3)
        ax.axvline(x=0, color='k', linewidth=2)

        # 圆柱
        cylinder = Circle((cx, cy), R_cylinder, facecolor='dimgray',
                          edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(cylinder)

        # 弹性板
        x_flag, y_flag = get_flag_shape(t_phase, n_points=100)
        ax.fill_between(x_flag, y_flag - flag_thickness/2, y_flag + flag_thickness/2,
                        color='orange', edgecolor='darkorange', linewidth=2, zorder=10)

        # 力箭头和压力区标注
        if idx == 0:  # 向上
            ax.arrow(0.55, 0.2, 0, 0.08, head_width=0.02, head_length=0.015,
                    fc='limegreen', ec='darkgreen', linewidth=2, zorder=15)
            ax.text(0.58, 0.24, 'F升', fontsize=12, color='darkgreen', fontweight='bold')
            ax.text(0.75, 0.35, '低压区', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))
            ax.text(0.75, 0.08, '高压区', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))
        else:  # 向下
            ax.arrow(0.55, 0.2, 0, -0.08, head_width=0.02, head_length=0.015,
                    fc='limegreen', ec='darkgreen', linewidth=2, zorder=15)
            ax.text(0.58, 0.16, 'F升', fontsize=12, color='darkgreen', fontweight='bold')
            ax.text(0.75, 0.35, '高压区', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))
            ax.text(0.75, 0.08, '低压区', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))

        # 标注
        ax.text(0.02, 0.02, '颜色: 速度大小\n流线: 流动方向',
               fontsize=8, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(-0.02, H_channel + 0.02)
        ax.set_aspect('equal')
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel('x (m)', fontsize=11)
        ax.set_ylabel('y (m)', fontsize=11)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('速度 |u| (m/s)', fontsize=10)

    plt.suptitle('Turek-Hron FSI2：涡激振动机理（卡门涡街交替脱落）', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fsi2_vortex_street(save_path):
    """绘制完整的涡街演化序列 - 清晰展示涡向下游移动"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 使用更长的时间跨度，让涡移动更明显
    # 涡对流速度约0.8 m/s，一个周期内移动约0.4m
    times = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0.5秒 = 一个周期
    titles = ['t = 0', 't = 0.1s', 't = 0.2s', 't = 0.3s', 't = 0.4s', 't = 0.5s (≈T)']

    # 涡的参数 - 用于追踪
    convection_speed = 0.8  # m/s
    vortex_spacing = 0.12
    lateral_spacing = 0.07

    for idx, (ax, t, title) in enumerate(zip(axes.flat, times, titles)):
        # 生成速度场
        X, Y, U, V, speed = generate_velocity_field(t, nx=80, ny=40)

        # 绘制速度大小云图
        levels = np.linspace(0, 1.8, 19)
        cf = ax.contourf(X, Y, speed, levels=levels, cmap='coolwarm', extend='max', alpha=0.85)

        # 绘制流线
        strm = ax.streamplot(X, Y, U, V, color='white', density=1.0, linewidth=0.5,
                             arrowsize=0.5, arrowstyle='->', broken_streamlines=True)

        # 通道边界
        ax.plot([0, 1.2], [0, 0], 'k-', linewidth=2)
        ax.plot([0, 1.2], [H_channel, H_channel], 'k-', linewidth=2)

        # 圆柱
        cylinder = Circle((cx, cy), R_cylinder, facecolor='dimgray',
                          edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(cylinder)

        # 弹性板
        x_flag, y_flag = get_flag_shape(t, n_points=100)
        ax.fill_between(x_flag, y_flag - flag_thickness/2, y_flag + flag_thickness/2,
                        color='orange', edgecolor='darkorange', linewidth=2, zorder=10)

        # 标记涡心位置 - 用圆圈和箭头清晰显示
        shedding_period = vortex_spacing / convection_speed
        for i in range(8):
            shed_time = i * shedding_period / 2
            vortex_x = 0.32 + convection_speed * (t - shed_time)

            if vortex_x < 0.30 or vortex_x > 0.95:
                continue

            if i % 2 == 0:
                vortex_y = cy + lateral_spacing
                color = 'red'
                marker = '↺'  # 逆时针
            else:
                vortex_y = cy - lateral_spacing
                color = 'blue'
                marker = '↻'  # 顺时针

            # 绘制涡心标记
            circle = Circle((vortex_x, vortex_y), 0.025, facecolor='none',
                           edgecolor=color, linewidth=2, linestyle='-', zorder=15)
            ax.add_patch(circle)

            # 添加旋转方向标记
            ax.text(vortex_x, vortex_y, marker, fontsize=10, color=color,
                   ha='center', va='center', fontweight='bold', zorder=16)

        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, H_channel + 0.02)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('速度 |u| (m/s)', fontsize=10)

    # 添加说明
    fig.text(0.5, 0.02, '红圈↺: 逆时针涡 (上排)    蓝圈↻: 顺时针涡 (下排)    涡对流速度 ≈ 0.8 m/s',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Turek-Hron FSI2：涡街演化与固体振动耦合 (涡随时间向下游移动)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 0.91, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figs_chap05"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Turek-Hron FSI2 visualization figures...")
    print("=" * 60)

    print("\n[1] FSI2 振动快照序列")
    plot_fsi2_snapshots(f"{output_dir}/turek_fsi2_snapshots.pdf")

    print("\n[2] FSI2 时间历程与分析")
    plot_fsi2_time_history(f"{output_dir}/turek_fsi2_time_history.pdf")

    print("\n[3] FSI2 涡激振动机理（含涡量场）")
    plot_fsi2_mechanism(f"{output_dir}/turek_fsi2_mechanism.pdf")

    print("\n[4] FSI2 涡街演化序列")
    plot_fsi2_vortex_street(f"{output_dir}/turek_fsi2_vortex_street.pdf")

    print("\n" + "=" * 60)
    print("All FSI2 figures generated!")
    print("=" * 60)
