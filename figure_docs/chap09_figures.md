# 第9章 图片说明

## activation_functions.pdf

**背景**：第9章介绍神经网络基础，激活函数是核心组件。

**意图**：对比不同激活函数的形状和特性：Sigmoid、Tanh、ReLU及其变体。

**生成方法**：
- 代码文件：`code_chap09/activation_functions.py`
- 绘制函数曲线和导数曲线
- 标注关键特性（饱和区、死区等）

**数据来源**：
- 解析函数定义
- x范围：[-5, 5]
- 标注梯度消失/爆炸区域

---

## modern_activations.pdf

**背景**：现代深度学习使用更先进的激活函数。

**意图**：介绍GELU、Swish、Mish等新型激活函数。

**生成方法**：
- 代码文件：`code_chap09/activation_functions.py`
- 对比新旧激活函数
- 展示平滑性差异

**数据来源**：
- GELU: x·Φ(x)
- Swish: x·sigmoid(x)
- Mish: x·tanh(softplus(x))

---

## computation_graph.pdf

**背景**：自动微分是深度学习的基础。

**意图**：用计算图解释反向传播的工作原理。

**生成方法**：
- 代码文件：`code_chap09/chap09_autograd.py`
- 使用networkx绘制计算图
- 标注前向值和反向梯度

**数据来源**：
- 示例表达式：f = (x + y) * z
- 手工计算的梯度值

---

## gradient_flow.pdf

**背景**：梯度流动决定了网络的可训练性。

**意图**：可视化梯度在深层网络中的传播，解释梯度消失/爆炸。

**生成方法**：
- 代码文件：`code_chap09/chap09_autograd.py`
- 构建多层网络
- 统计各层梯度范数

**数据来源**：
- 10层全连接网络
- 随机初始化权重
- 比较Sigmoid vs ReLU

---

## gradient_check.pdf

**背景**：数值梯度检验是调试的重要工具。

**意图**：展示如何验证自动微分的正确性。

**生成方法**：
- 代码文件：`code_chap09/chap09_autograd.py`
- 对比解析梯度和数值梯度
- 绘制相对误差

**数据来源**：
- 有限差分步长：ε = 1e-5
- 相对误差 = |∂f/∂x - Δf/Δx| / max(|∂f/∂x|, |Δf/Δx|)
