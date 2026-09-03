# 第4章 图片说明

## discrete_to_continuum.pdf

**背景**：第4章讲解从离散系统到连续介质的过渡。

**意图**：展示弹簧-质点链在极限情况下如何过渡到连续弹性杆，理解PDE的物理起源。

**生成方法**：
- 代码文件：`code_chap04/continuum_pde_visualization.py`
- 绘制N个质点的弹簧链
- 逐步增加N，展示趋近连续的过程

**数据来源**：
- 离散系统：N = 5, 10, 20, 50个质点
- 连续极限：波动方程 ∂²u/∂t² = c²∂²u/∂x²

---

## heat_equation_visualization.pdf

**背景**：热传导方程是抛物型PDE的典型代表。

**意图**：可视化热量如何在杆中扩散，展示热方程的耗散特性。

**生成方法**：
- 代码文件：`code_chap04/continuum_pde_visualization.py`
- 有限差分法求解1D热方程
- 绘制温度场随时间演化的热图

**数据来源**：
- 初始条件：δ函数（点热源）
- 边界条件：Dirichlet（两端固定温度）
- 扩散系数：α = 0.01

---

## wave_equation_visualization.pdf

**背景**：波动方程是双曲型PDE的典型代表。

**意图**：展示波的传播、反射现象，对比热方程的耗散行为。

**生成方法**：
- 代码文件：`code_chap04/continuum_pde_visualization.py`
- 有限差分法求解1D波方程
- 绘制波形随时间传播的动画快照

**数据来源**：
- 初始条件：高斯波包
- 边界条件：固定端（Dirichlet）
- 波速：c = 1

---

## poisson_equation_visualization.pdf

**背景**：Poisson方程是椭圆型PDE的典型代表。

**意图**：展示稳态问题的求解，如静电场、稳态温度场。

**生成方法**：
- 代码文件：`code_chap04/continuum_pde_visualization.py`
- 有限差分法求解2D Poisson方程
- 绘制解的等高线图和3D表面

**数据来源**：
- 源项：点源或分布源
- 边界条件：Dirichlet
- 网格：50×50

---

## pde_comparison.pdf

**背景**：对比三类PDE的特性。

**意图**：让读者一目了然地理解椭圆、抛物、双曲方程的本质区别。

**生成方法**：
- 代码文件：`code_chap04/continuum_pde_visualization.py`
- 并排绘制三种PDE的解
- 标注各自特征（耗散/传播/稳态）

**数据来源**：
- 使用相同的空间域和初始条件
- 展示相同时刻的解

---

## pendulum_comparison.pdf

**背景**：单摆是ODE数值方法的经典测试问题。

**意图**：比较不同数值方法（Euler、RK4、辛方法）的长时间稳定性。

**生成方法**：
- 代码文件：`code_chap04/ode_numerical_methods.py`
- 求解单摆方程 θ'' + (g/L)sinθ = 0
- 绘制相空间轨迹

**数据来源**：
- 摆长L=1m，g=9.8m/s²
- 初始角度θ₀=π/4
- 长时间积分：t=0到100s

---

## convergence_order.pdf

**背景**：数值方法的收敛阶是重要的理论性质。

**意图**：验证各方法的理论收敛阶（Euler: 1阶, RK4: 4阶）。

**生成方法**：
- 代码文件：`code_chap04/ode_numerical_methods.py`
- 变化步长h，计算误差
- 对数坐标绑图验证斜率

**数据来源**：
- 有解析解的测试问题
- 步长h从0.1到0.001
- 误差 = |数值解 - 解析解|

---

## harmonic_comparison.pdf

**背景**：简谐振子是辛方法的理想测试问题。

**意图**：展示辛方法（如蛙跳法）如何保持能量守恒。

**生成方法**：
- 代码文件：`code_chap04/ode_numerical_methods.py`
- 求解 x'' + x = 0
- 对比Euler和辛Euler的能量变化

**数据来源**：
- 初始条件：x(0)=1, x'(0)=0
- 解析解：x = cos(t)
- 能量：E = (x² + v²)/2

---

## long_time_comparison.pdf

**背景**：长时间积分是数值方法的关键挑战。

**意图**：展示为什么天体力学等问题需要辛积分器。

**生成方法**：
- 代码文件：`code_chap04/ode_numerical_methods.py`
- 积分到t=1000
- 对比能量漂移

**数据来源**：
- Kepler问题或简谐振子
- 长时间能量误差统计
