# 第13章 图片说明

## pinn_poisson_1d.pdf / pinn_poisson_1d.png

**背景**：第13章介绍物理信息神经网络（PINN）。

**意图**：展示PINN求解1D Poisson方程的效果，对比解析解。

**生成方法**：
- 代码文件：`code_chap13/pinn_poisson_1d.py`
- PyTorch实现PINN
- 绘制预测解、真实解、误差

**数据来源**：
- 方程：-u'' = f(x)，x∈[0,1]
- 边界条件：u(0)=u(1)=0
- 解析解用于验证

---

## pinn_poisson_2d.pdf / pinn_poisson_2d.png

**背景**：2D问题展示PINN的实用价值。

**意图**：可视化2D Poisson方程的PINN解。

**生成方法**：
- 代码文件：`code_chap13/pinn_poisson_2d.py`
- 2D神经网络输出
- 等高线图和3D表面

**数据来源**：
- 方程：-Δu = f(x,y)，(x,y)∈[0,1]²
- 网格：50×50配点
- 训练迭代：10000

---

## pinn_poisson_2d_slices.pdf / pinn_poisson_2d_slices.png

**背景**：切片视图帮助理解2D解的细节。

**意图**：展示固定x或y时的解剖面。

**生成方法**：
- 代码文件：`code_chap13/pinn_poisson_2d.py`
- 沿x=0.5和y=0.5切片
- 对比PINN解和参考解

**数据来源**：
- 同上2D问题
- 切片位置：中线

---

## pinn_workflow.pdf / pinn_workflow.png

**背景**：PINN的工作流程需要清晰说明。

**意图**：图解PINN的训练流程：采样→网络前向→计算残差→反向传播。

**生成方法**：
- 代码文件：`code_chap13/regenerate_chinese_figures.py`
- 流程图绘制
- 中文标注

**数据来源**：
- 概念性示意图
- 无具体数值数据

---

## spectral_bias.pdf / spectral_bias.png

**背景**：谱偏差是PINN的重要挑战。

**意图**：展示神经网络倾向于先学低频成分。

**生成方法**：
- 代码文件：`code_chap13/spectral_bias.py`
- 训练PINN拟合高频函数
- 绘制不同训练阶段的频谱

**数据来源**：
- 目标函数：sin(kx)，k=1,5,10,20
- 记录训练过程中的频谱分解

---

## fourier_features.pdf / fourier_features.png

**背景**：Fourier特征可缓解谱偏差。

**意图**：对比普通PINN和Fourier feature PINN的高频学习能力。

**生成方法**：
- 代码文件：`code_chap13/fourier_features.py`
- 实现随机Fourier特征映射
- 对比学习曲线

**数据来源**：
- 特征维度：256
- 频率采样：高斯分布

---

## lambda_effect.pdf / lambda_effect.png

**背景**：损失函数权重λ影响训练效果。

**意图**：展示不同λ值下PDE损失与边界损失的平衡。

**生成方法**：
- 代码文件：`code_chap13/lambda_effect.py`
- 扫描λ从0.1到100
- 绘制最终误差vs λ

**数据来源**：
- 总损失 = L_pde + λ·L_bc
- 记录不同λ的收敛结果

---

## adaptive_weight.pdf / adaptive_weight.png

**背景**：自适应权重可自动平衡多任务损失。

**意图**：展示自适应方法如何动态调整权重。

**生成方法**：
- 代码文件：`code_chap13/adaptive_weight.py`
- 实现梯度归一化方法
- 绘制权重随训练的变化

**数据来源**：
- 训练过程中记录的权重轨迹
- 对比固定权重和自适应权重

---

## gradient_diagnosis.pdf / gradient_diagnosis.png

**背景**：梯度病态是PINN训练困难的原因之一。

**意图**：可视化不同损失项的梯度范数差异。

**生成方法**：
- 代码文件：`code_chap13/gradient_diagnosis.py`
- 计算各损失项的梯度
- 柱状图对比

**数据来源**：
- 记录训练过程中的梯度统计
- ||∇L_pde|| vs ||∇L_bc|| vs ||∇L_data||

---

## hard_vs_soft.pdf / hard_vs_soft.png

**背景**：硬约束vs软约束是PINN的重要设计选择。

**意图**：对比两种边界条件处理方式的效果。

**生成方法**：
- 代码文件：`code_chap13/hard_constraint.py`
- 软约束：惩罚项
- 硬约束：网络结构设计

**数据来源**：
- 硬约束：u_nn = x(1-x)·N(x)
- 软约束：u_nn = N(x)，加边界惩罚

---

## constraint_concept.pdf / constraint_concept.png

**背景**：解释约束的概念。

**意图**：示意图说明硬约束如何嵌入网络结构。

**生成方法**：
- 代码文件：`code_chap13/regenerate_chinese_figures.py`
- 概念图绘制

**数据来源**：
- 概念性示意图

---

## deep_ritz_comparison.pdf / deep_ritz_comparison.png

**背景**：Deep Ritz方法是PINN的变体。

**意图**：对比PINN（强形式）和Deep Ritz（弱形式）的收敛性。

**生成方法**：
- 代码文件：`code_chap13/additional_experiments.py`
- 实现两种方法
- 绘制收敛曲线对比

**数据来源**：
- 相同的PDE问题
- 相同的网络结构
- 对比训练损失和测试误差

---

## inverse_problem.pdf / inverse_problem.png

**背景**：PINN可用于参数辨识。

**意图**：展示如何从观测数据反演PDE参数。

**生成方法**：
- 代码文件：`code_chap13/additional_experiments.py`
- 设置逆问题：已知解，求系数
- 绘制参数收敛过程

**数据来源**：
- 真实参数：k=1.0（待辨识）
- 观测数据：带噪声的解
- 辨识结果：k_pred随迭代的变化
