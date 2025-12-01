# 第5章 图片说明

## fsi_mesh_setup.pdf

**背景**：第5章介绍流固耦合（FSI），需要展示计算域和网格设置。

**意图**：说明FSI问题的几何设置：流体域、固体域、交界面。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 使用matplotlib绑制示意图
- 标注边界条件类型

**数据来源**：
- 悬臂梁在流场中的标准设置
- 几何参数参考Turek-Hron基准

---

## fsi_fem_matrices.pdf

**背景**：FSI的数值实现涉及耦合矩阵组装。

**意图**：可视化流体、固体、耦合子矩阵的稀疏结构。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 组装简化的FEM矩阵
- 使用spy()绘制稀疏模式

**数据来源**：
- 简化的2D FSI模型
- 流体：Stokes方程
- 固体：线弹性

---

## fsi_modal_shapes.pdf

**背景**：固体的振动模态影响FSI响应。

**意图**：展示悬臂梁的前几阶弯曲模态。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 求解特征值问题
- 绘制模态振型

**数据来源**：
- Euler-Bernoulli梁理论
- 解析模态：φₙ(x) = cosh(βₙx) - cos(βₙx) - ...

---

## fsi_time_response.pdf

**背景**：FSI的瞬态响应是关键输出。

**意图**：展示梁端点位移随时间的变化，观察振荡和衰减。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 时间推进求解耦合系统
- 绘制位移-时间曲线

**数据来源**：
- 时间步长：Δt = 0.001s
- 总时间：T = 10s
- 初始条件：静止

---

## fsi_velocity_sweep.pdf

**背景**：来流速度是FSI的关键参数。

**意图**：展示不同流速下的响应幅值，识别共振区域。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 扫描流速从0.1到10 m/s
- 记录稳态振幅

**数据来源**：
- 流速扫描：20个点
- 每个流速运行到稳态

---

## fsi_deformation_snapshots.pdf

**背景**：可视化FSI的动态变形过程。

**意图**：展示不同时刻的梁变形和流场。

**生成方法**：
- 代码文件：`code_chap05/fsi_cantilever_beam.py`
- 选取典型时刻的快照
- 叠加绘制变形和流线

**数据来源**：
- 时刻选取：t = 0, T/4, T/2, 3T/4, T
- T为主振荡周期

---

## turek_fsi2_cfd_velocity.pdf

**背景**：Turek-Hron FSI2是国际公认的FSI基准问题。

**意图**：展示真实CFD模拟的速度场演化，验证涡街形成。

**生成方法**：
- 数据来源：Kratos Multiphysics开源CFD软件
- 从官方GitHub仓库下载预计算结果
- 提取GIF动画的关键帧
- 组合成PDF

**数据来源**：
- Kratos官方FSI2验证案例
- URL: https://github.com/KratosMultiphysics/Examples
- Re = 100，弹性旗帜

---

## turek_fsi2_mechanism.pdf

**背景**：解释FSI2中的物理机制。

**意图**：图解涡脱落如何激励固体振动。

**生成方法**：
- 代码文件：`code_chap05/turek_fsi2_visualization.py`
- 示意图绘制
- 标注涡、升力、位移的关系

**数据来源**：
- Turek-Hron论文几何参数
- 圆柱直径D=0.1m，旗帜长度L=0.35m

---

## turek_fsi2_snapshots.pdf

**背景**：展示FSI2的变形序列。

**意图**：直观展示旗帜在流场中的摆动。

**生成方法**：
- 代码文件：`code_chap05/turek_fsi2_visualization.py`
- 参数化旗帜变形
- 叠加流线绘制

**数据来源**：
- 简化的正弦模态变形
- 振幅参考Turek-Hron结果

---

## turek_fsi2_time_history.pdf

**背景**：FSI2的定量验证需要时间历程。

**意图**：展示旗帜端点位移的周期性振荡。

**生成方法**：
- 代码文件：`code_chap05/turek_fsi2_visualization.py`
- 绘制y方向位移vs时间
- 标注振幅和频率

**数据来源**：
- Turek-Hron参考值：
  - 频率 f ≈ 2 Hz
  - 振幅 A ≈ 0.08m

---

## turek_fsi2_vortex_street.pdf

**背景**：卡门涡街是FSI的驱动力。

**意图**：可视化涡度场，展示涡脱落的规律性。

**生成方法**：
- 代码文件：`code_chap05/turek_fsi2_visualization.py`
- Rankine涡模型
- 涡度等高线绘制

**数据来源**：
- 简化的涡街模型
- Strouhal数 St ≈ 0.2
