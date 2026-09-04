"""
第13章扩展：流固耦合有限元示例
模拟高速流体吹动悬臂梁的振动响应

物理模型：
- 结构：悬臂梁（一端固定，一端自由）
- 流体：横向均匀来流，产生涡激振动
- 耦合：流体力作用于结构，结构变形影响流场

简化模型采用：
1. 结构：Euler-Bernoulli梁有限元
2. 流体力：准稳态涡脱落模型（Fung振子模型）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 结构参数（悬臂梁）
# =============================================================================
L = 1.0          # 梁长度 (m)
b = 0.05         # 梁宽度 (m)
h = 0.02         # 梁厚度 (m)
E = 2.1e11       # 弹性模量 (Pa) - 钢
rho_s = 7850     # 结构密度 (kg/m³)
I = b * h**3 / 12  # 截面惯性矩
A = b * h        # 截面面积
m_per_length = rho_s * A  # 单位长度质量

# 流体参数
rho_f = 1.225    # 流体密度 (kg/m³) - 空气
U_inf = 20.0     # 来流速度 (m/s)
D = h            # 特征尺寸（梁厚度）
St = 0.2         # Strouhal数（涡脱落频率特征）
f_s = St * U_inf / D  # 涡脱落频率

# 阻尼参数
zeta = 0.01      # 阻尼比

# =============================================================================
# 有限元模型：Euler-Bernoulli梁
# =============================================================================
def beam_element_matrices(Le, EI, m):
    """
    计算单个梁单元的刚度矩阵和质量矩阵

    参数：
        Le: 单元长度
        EI: 抗弯刚度
        m: 单位长度质量

    返回：
        Ke: 4x4 单元刚度矩阵
        Me: 4x4 单元质量矩阵
    """
    # 刚度矩阵
    Ke = EI / Le**3 * np.array([
        [12, 6*Le, -12, 6*Le],
        [6*Le, 4*Le**2, -6*Le, 2*Le**2],
        [-12, -6*Le, 12, -6*Le],
        [6*Le, 2*Le**2, -6*Le, 4*Le**2]
    ])

    # 一致质量矩阵
    Me = m * Le / 420 * np.array([
        [156, 22*Le, 54, -13*Le],
        [22*Le, 4*Le**2, 13*Le, -3*Le**2],
        [54, 13*Le, 156, -22*Le],
        [-13*Le, -3*Le**2, -22*Le, 4*Le**2]
    ])

    return Ke, Me

def assemble_global_matrices(n_elements, L_total, EI, m):
    """
    组装全局刚度和质量矩阵
    """
    Le = L_total / n_elements
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes  # 每个节点2个自由度：挠度w和转角θ

    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    for e in range(n_elements):
        Ke, Me = beam_element_matrices(Le, EI, m)

        # 单元自由度索引
        dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]

        # 组装
        for i in range(4):
            for j in range(4):
                K[dof_indices[i], dof_indices[j]] += Ke[i, j]
                M[dof_indices[i], dof_indices[j]] += Me[i, j]

    return K, M, Le

def apply_boundary_conditions(K, M, fixed_dofs):
    """
    应用边界条件（消去法）
    """
    n_dof = K.shape[0]
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]

    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    M_reduced = M[np.ix_(free_dofs, free_dofs)]

    return K_reduced, M_reduced, free_dofs

def modal_analysis(K, M):
    """
    模态分析：求解特征值问题
    """
    eigenvalues, eigenvectors = eigh(K, M)
    natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Hz
    return natural_frequencies, eigenvectors

# =============================================================================
# 流体力模型（Fung振子模型简化版）
# =============================================================================
def fluid_force(t, w_tip, w_dot_tip, U, D, rho_f, L, b):
    """
    计算涡激振动产生的流体力

    使用半经验模型：
    F = F_lift + F_drag_fluctuation

    其中升力项与涡脱落相关
    """
    # 涡脱落角频率
    omega_s = 2 * np.pi * St * U / D

    # 升力系数振荡
    CL_0 = 0.5  # 基础升力系数幅值
    CL = CL_0 * np.sin(omega_s * t)

    # 动压
    q = 0.5 * rho_f * U**2

    # 升力（作用于整个梁的等效集中力）
    F_lift = q * D * L * CL

    # 附加阻尼（流体阻尼）
    C_fluid = 0.1 * rho_f * U * D * L
    F_damping = -C_fluid * w_dot_tip

    return F_lift + F_damping

# =============================================================================
# 时域仿真
# =============================================================================
def simulate_fsi(n_elements=10, t_end=5.0, U=20.0):
    """
    流固耦合时域仿真
    """
    # 组装矩阵
    EI = E * I
    K_global, M_global, Le = assemble_global_matrices(n_elements, L, EI, m_per_length)

    # 边界条件：固定端（x=0）的挠度和转角
    fixed_dofs = [0, 1]
    K, M, free_dofs = apply_boundary_conditions(K_global, M_global, fixed_dofs)

    n_dof = len(free_dofs)

    # 模态分析
    freqs, modes = modal_analysis(K, M)
    omega_n = 2 * np.pi * freqs  # 角频率

    # 阻尼矩阵（Rayleigh阻尼）
    alpha = 2 * zeta * omega_n[0] * omega_n[1] / (omega_n[0] + omega_n[1])
    beta = 2 * zeta / (omega_n[0] + omega_n[1])
    C = alpha * M + beta * K

    # 力分布向量（集中力作用在自由端）
    F_vec = np.zeros(n_dof)
    F_vec[-2] = 1.0  # 自由端挠度对应的自由度

    # 状态空间方程：[w; w_dot]' = [0, I; -M^{-1}K, -M^{-1}C][w; w_dot] + [0; M^{-1}F]
    M_inv = np.linalg.inv(M)

    def dynamics(t, state):
        w = state[:n_dof]
        w_dot = state[n_dof:]

        # 自由端位移和速度
        w_tip = w[-2]
        w_dot_tip = w_dot[-2]

        # 流体力
        F_fluid = fluid_force(t, w_tip, w_dot_tip, U, D, rho_f, L, b)
        F = F_vec * F_fluid

        # 运动方程
        w_ddot = M_inv @ (F - C @ w_dot - K @ w)

        return np.concatenate([w_dot, w_ddot])

    # 初始条件
    state0 = np.zeros(2 * n_dof)

    # 时间积分
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)

    sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, method='RK45',
                   rtol=1e-6, atol=1e-9)

    return sol.t, sol.y[:n_dof, :], freqs, modes, free_dofs, Le, n_elements

# =============================================================================
# 可视化函数
# =============================================================================
def plot_mesh_and_setup(save_path):
    """
    绘制有限元网格和问题设置
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：物理模型示意
    ax1 = axes[0]
    ax1.set_xlim(-0.3, 1.5)
    ax1.set_ylim(-0.4, 0.4)

    # 绘制地面
    ax1.fill_between([-0.3, 0], [-0.4, -0.4], [0.4, 0.4], color='gray', alpha=0.3)
    ax1.axvline(x=0, color='black', linewidth=2)

    # 绘制梁
    beam = Rectangle((0, -0.02), 1.0, 0.04, fill=True, color='steelblue',
                     edgecolor='black', linewidth=1.5)
    ax1.add_patch(beam)

    # 绘制变形后的梁（虚线）
    x_beam = np.linspace(0, 1, 50)
    y_deform = 0.1 * (x_beam/1.0)**2 * np.sin(2*np.pi*0.3)  # 示意变形
    ax1.plot(x_beam, y_deform, 'b--', linewidth=2, alpha=0.5, label='Deformed shape')

    # 绘制流动箭头
    for y in np.linspace(-0.3, 0.3, 7):
        ax1.annotate('', xy=(0.8, y), xytext=(0.3, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.text(0.55, 0.35, 'Fluid flow\n$U_\\infty$', fontsize=12, ha='center', color='red')

    # 标注
    ax1.annotate('Fixed end', xy=(0, 0), xytext=(-0.25, 0.2),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('Free end\n(tip)', xy=(1.0, 0), xytext=(1.2, 0.2),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # 尺寸标注
    ax1.annotate('', xy=(0, -0.15), xytext=(1.0, -0.15),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax1.text(0.5, -0.22, 'L = 1.0 m', fontsize=11, ha='center')

    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Physical Model: Cantilever Beam in Cross-Flow', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 右图：有限元网格
    ax2 = axes[1]
    n_elem = 8
    Le = 1.0 / n_elem

    # 绘制单元
    for i in range(n_elem):
        x_start = i * Le
        x_end = (i + 1) * Le
        ax2.plot([x_start, x_end], [0, 0], 'b-', linewidth=3)
        ax2.plot(x_start, 0, 'ko', markersize=10)
        ax2.text(x_start, -0.08, f'$n_{i}$', fontsize=10, ha='center')
    ax2.plot(1.0, 0, 'ko', markersize=10)
    ax2.text(1.0, -0.08, f'$n_{n_elem}$', fontsize=10, ha='center')

    # 标注单元
    for i in range(n_elem):
        x_mid = (i + 0.5) * Le
        ax2.text(x_mid, 0.05, f'$e_{i+1}$', fontsize=10, ha='center', color='blue')

    # 自由度标注
    ax2.annotate('DOF: $w_i, \\theta_i$', xy=(0.5, 0), xytext=(0.5, 0.2),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # 边界条件
    ax2.plot([0, 0], [-0.03, 0.03], 'k-', linewidth=8)
    ax2.text(0, 0.12, 'BC: $w_0=0$\n$\\theta_0=0$', fontsize=10, ha='center')

    ax2.set_xlim(-0.1, 1.2)
    ax2.set_ylim(-0.25, 0.35)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_title('FEM Mesh: Euler-Bernoulli Beam Elements', fontsize=13)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_modal_shapes(save_path):
    """
    绘制前几阶模态振型
    """
    n_elements = 20
    EI = E * I
    K_global, M_global, Le = assemble_global_matrices(n_elements, L, EI, m_per_length)

    fixed_dofs = [0, 1]
    K, M, free_dofs = apply_boundary_conditions(K_global, M_global, fixed_dofs)

    freqs, modes = modal_analysis(K, M)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x_nodes = np.linspace(0, L, n_elements + 1)

    for idx, ax in enumerate(axes.flat):
        if idx < 4:
            # 提取挠度自由度的模态振型
            mode_shape = np.zeros(n_elements + 1)
            mode_shape[0] = 0  # 固定端
            for i, dof in enumerate(free_dofs):
                if dof % 2 == 0:  # 挠度自由度
                    node_idx = dof // 2
                    mode_shape[node_idx] = modes[i, idx]

            # 归一化
            mode_shape = mode_shape / np.max(np.abs(mode_shape)) if np.max(np.abs(mode_shape)) > 0 else mode_shape

            ax.plot(x_nodes, mode_shape, 'b-', linewidth=2.5)
            ax.fill_between(x_nodes, mode_shape, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.plot(x_nodes, np.zeros_like(x_nodes), 'k--', linewidth=1, alpha=0.5)

            ax.set_xlabel('x (m)', fontsize=11)
            ax.set_ylabel('Normalized displacement', fontsize=11)
            ax.set_title(f'Mode {idx+1}: $f_{idx+1}$ = {freqs[idx]:.2f} Hz', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, L)

    plt.suptitle('Natural Mode Shapes of Cantilever Beam', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_time_response(save_path):
    """
    绘制时域响应
    """
    t, w_history, freqs, modes, free_dofs, Le, n_elements = simulate_fsi(n_elements=10, t_end=5.0, U=20.0)

    # 提取自由端挠度
    tip_idx = -2  # 最后一个节点的挠度自由度
    w_tip = w_history[tip_idx, :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 时间历程
    ax1 = axes[0, 0]
    ax1.plot(t, w_tip * 1000, 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Tip displacement (mm)', fontsize=11)
    ax1.set_title('Time History of Tip Displacement', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # 相图
    ax2 = axes[0, 1]
    w_dot_tip = np.gradient(w_tip, t)
    ax2.plot(w_tip[100:] * 1000, w_dot_tip[100:] * 1000, 'b-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Displacement (mm)', fontsize=11)
    ax2.set_ylabel('Velocity (mm/s)', fontsize=11)
    ax2.set_title('Phase Portrait (after transient)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 频谱分析
    ax3 = axes[1, 0]
    from scipy.fft import fft, fftfreq
    N = len(t)
    dt = t[1] - t[0]
    yf = np.abs(fft(w_tip))[:N//2]
    xf = fftfreq(N, dt)[:N//2]

    ax3.semilogy(xf, yf * 1000, 'b-', linewidth=1)
    ax3.axvline(x=freqs[0], color='r', linestyle='--', label=f'$f_1$ = {freqs[0]:.1f} Hz')
    ax3.axvline(x=f_s, color='g', linestyle='--', label=f'$f_s$ = {f_s:.1f} Hz (vortex)')
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('Amplitude (mm)', fontsize=11)
    ax3.set_title('Frequency Spectrum', fontsize=12)
    ax3.set_xlim(0, 100)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 不同流速响应
    ax4 = axes[1, 1]
    velocities = [5, 10, 20, 30]
    colors = ['blue', 'green', 'orange', 'red']

    for U, color in zip(velocities, colors):
        t_sim, w_sim, _, _, _, _, _ = simulate_fsi(n_elements=10, t_end=3.0, U=U)
        w_tip_sim = w_sim[-2, :]
        # 取稳态后的最大振幅
        steady_amp = np.max(np.abs(w_tip_sim[len(t_sim)//2:])) * 1000
        ax4.plot(t_sim, w_sim[-2, :] * 1000, color=color, linewidth=1,
                label=f'U = {U} m/s, A = {steady_amp:.2f} mm', alpha=0.8)

    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Tip displacement (mm)', fontsize=11)
    ax4.set_title('Response at Different Flow Velocities', fontsize=12)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_beam_animation_frames(save_path):
    """
    绘制梁变形的几个时刻快照
    """
    t, w_history, freqs, modes, free_dofs, Le, n_elements = simulate_fsi(n_elements=15, t_end=2.0, U=20.0)

    # 选择几个时刻
    time_indices = [int(len(t)*0.3), int(len(t)*0.4), int(len(t)*0.5),
                   int(len(t)*0.6), int(len(t)*0.7), int(len(t)*0.8)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    x_nodes = np.linspace(0, L, n_elements + 1)

    for ax, t_idx in zip(axes.flat, time_indices):
        # 重建变形形状
        w_shape = np.zeros(n_elements + 1)
        w_shape[0] = 0  # 固定端

        for i, dof in enumerate(free_dofs):
            if dof % 2 == 0:  # 挠度自由度
                node_idx = dof // 2
                if node_idx < n_elements + 1:
                    w_shape[node_idx] = w_history[i, t_idx]

        # 放大变形以便可视化
        scale = 50
        y_deformed = w_shape * scale

        # 绘制未变形梁
        ax.plot(x_nodes, np.zeros_like(x_nodes), 'k--', linewidth=1, alpha=0.5, label='Undeformed')

        # 绘制变形梁
        ax.plot(x_nodes, y_deformed, 'b-', linewidth=3, label='Deformed')
        ax.fill_between(x_nodes, y_deformed - 0.005, y_deformed + 0.005,
                       color='steelblue', alpha=0.5)

        # 绘制节点
        ax.plot(x_nodes, y_deformed, 'ko', markersize=4)

        # 固定端
        ax.plot([0, 0], [-0.05, 0.05], 'k-', linewidth=6)

        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (scaled)', fontsize=10)
        ax.set_title(f't = {t[t_idx]:.3f} s', fontsize=11)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 0.15)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Beam Deformation at Different Time Instants (50x magnification)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_velocity_sweep(save_path):
    """
    绘制流速扫描响应（涡激振动锁定现象）
    """
    velocities = np.linspace(5, 40, 8)  # 减少采样点加速计算
    amplitudes = []

    # 计算固有频率
    n_elements = 10
    EI = E * I
    K_global, M_global, Le = assemble_global_matrices(n_elements, L, EI, m_per_length)
    fixed_dofs = [0, 1]
    K, M, free_dofs = apply_boundary_conditions(K_global, M_global, fixed_dofs)
    freqs, _ = modal_analysis(K, M)
    f_n = freqs[0]  # 第一阶固有频率

    # 计算共振流速
    U_r = f_n * D / St  # 使涡脱落频率等于固有频率的流速

    print("Performing velocity sweep...")
    for U in velocities:
        t, w_history, _, _, _, _, _ = simulate_fsi(n_elements=10, t_end=2.0, U=U)  # 缩短时间
        w_tip = w_history[-2, :]
        # 稳态振幅
        amp = np.max(np.abs(w_tip[len(t)//2:])) * 1000
        amplitudes.append(amp)
        print(f"  U = {U:.1f} m/s, Amplitude = {amp:.3f} mm")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 振幅vs流速
    ax1.plot(velocities, amplitudes, 'bo-', markersize=8, linewidth=2)
    ax1.axvline(x=U_r, color='r', linestyle='--', linewidth=2,
               label=f'Resonance $U_r$ = {U_r:.1f} m/s')
    ax1.set_xlabel('Flow velocity $U$ (m/s)', fontsize=12)
    ax1.set_ylabel('Steady-state amplitude (mm)', fontsize=12)
    ax1.set_title('Vortex-Induced Vibration: Amplitude vs Velocity', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 约化速度
    U_reduced = velocities / (f_n * D)
    ax2.plot(U_reduced, amplitudes, 'go-', markersize=8, linewidth=2)
    ax2.axvline(x=1/St, color='r', linestyle='--', linewidth=2,
               label=f'Lock-in: $U_r^*$ = 1/St = {1/St:.1f}')
    ax2.set_xlabel('Reduced velocity $U^* = U/(f_n D)$', fontsize=12)
    ax2.set_ylabel('Steady-state amplitude (mm)', fontsize=12)
    ax2.set_title('Response in Terms of Reduced Velocity', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_fem_matrices(save_path):
    """
    可视化有限元矩阵的稀疏结构
    """
    n_elements = 6
    EI = E * I
    K, M, Le = assemble_global_matrices(n_elements, L, EI, m_per_length)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 刚度矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(np.log10(np.abs(K) + 1e-20), cmap='Blues', aspect='equal')
    ax1.set_title('Stiffness Matrix K (log scale)', fontsize=12)
    ax1.set_xlabel('DOF index', fontsize=11)
    ax1.set_ylabel('DOF index', fontsize=11)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 质量矩阵
    ax2 = axes[1]
    im2 = ax2.imshow(np.log10(np.abs(M) + 1e-20), cmap='Reds', aspect='equal')
    ax2.set_title('Mass Matrix M (log scale)', fontsize=12)
    ax2.set_xlabel('DOF index', fontsize=11)
    ax2.set_ylabel('DOF index', fontsize=11)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 稀疏模式
    ax3 = axes[2]
    ax3.spy(K, markersize=8, color='blue', marker='s')
    ax3.set_title('Sparsity Pattern of K', fontsize=12)
    ax3.set_xlabel('DOF index', fontsize=11)
    ax3.set_ylabel('DOF index', fontsize=11)

    plt.suptitle(f'FEM Matrices for {n_elements}-Element Beam', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    output_dir = "/home/user/full_book_claude/figures/chap06"

    print("=" * 60)
    print("FSI Cantilever Beam Simulation")
    print("=" * 60)

    # 打印系统参数
    print(f"\nStructural Parameters:")
    print(f"  Length L = {L} m")
    print(f"  Width b = {b} m")
    print(f"  Thickness h = {h} m")
    print(f"  Young's modulus E = {E/1e9:.1f} GPa")
    print(f"  Density rho_s = {rho_s} kg/m³")

    print(f"\nFluid Parameters:")
    print(f"  Density rho_f = {rho_f} kg/m³")
    print(f"  Flow velocity U = {U_inf} m/s")
    print(f"  Strouhal number St = {St}")
    print(f"  Vortex shedding frequency f_s = {f_s:.2f} Hz")

    # 计算固有频率
    n_elem = 10
    EI = E * I
    K_global, M_global, Le = assemble_global_matrices(n_elem, L, EI, m_per_length)
    fixed_dofs = [0, 1]
    K, M, free_dofs = apply_boundary_conditions(K_global, M_global, fixed_dofs)
    freqs, _ = modal_analysis(K, M)
    print(f"\nNatural Frequencies (first 4 modes):")
    for i in range(min(4, len(freqs))):
        print(f"  Mode {i+1}: f_{i+1} = {freqs[i]:.2f} Hz")

    # 生成图表
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    print("\n[1] Mesh and setup diagram")
    plot_mesh_and_setup(f"{output_dir}/fsi_mesh_setup.pdf")

    print("\n[2] Modal shapes")
    plot_modal_shapes(f"{output_dir}/fsi_modal_shapes.pdf")

    print("\n[3] Time response")
    plot_time_response(f"{output_dir}/fsi_time_response.pdf")

    print("\n[4] Deformation snapshots")
    plot_beam_animation_frames(f"{output_dir}/fsi_deformation_snapshots.pdf")

    print("\n[5] Velocity sweep")
    plot_velocity_sweep(f"{output_dir}/fsi_velocity_sweep.pdf")

    print("\n[6] FEM matrices visualization")
    plot_fem_matrices(f"{output_dir}/fsi_fem_matrices.pdf")

    print("\n" + "=" * 60)
    print("All FSI simulations completed!")
    print("=" * 60)
