# 第2章 图片说明

## lp_performance.pdf

**背景**：第2章介绍数值优化方法，需要展示线性规划求解器的性能。

**意图**：比较scipy.optimize.linprog在不同问题规模下的求解时间，让读者了解算法复杂度。

**生成方法**：
- 代码文件：`code_chap02/optimization_tools_demo.py`
- 随机生成不同规模的线性规划问题
- 使用scipy的HiGHS求解器
- 记录求解时间并绑图

**数据来源**：
- 随机生成的LP问题（固定种子保证可复现）
- 变量数从10到1000
- 约束数与变量数同阶

---

## qp_performance.pdf

**背景**：二次规划是机器学习中常见的优化问题（如SVM）。

**意图**：展示投资组合优化（典型QP问题）的求解性能随资产数量的变化。

**生成方法**：
- 代码文件：`code_chap02/optimization_tools_demo.py`
- 模拟Markowitz投资组合优化
- 最小化风险（二次项）+ 收益约束（线性约束）

**数据来源**：
- 随机生成的协方差矩阵（保证正定）
- 随机生成的预期收益向量
- 资产数从10到200

---

## portfolio_weights.pdf

**背景**：展示投资组合优化的实际结果。

**意图**：可视化最优投资权重分配，帮助读者理解QP求解结果的实际意义。

**生成方法**：
- 代码文件：`code_chap02/optimization_tools_demo.py`
- 求解50资产的投资组合优化
- 绑制权重柱状图

**数据来源**：
- 50个资产的模拟数据
- 目标收益率8%

---

## nonconvex_comparison.pdf

**背景**：非凸优化是深度学习的核心挑战。

**意图**：比较不同优化器（梯度下降、Adam、BFGS）在非凸函数上的表现。

**生成方法**：
- 代码文件：`code_chap02/optimization_tools_demo.py`
- 定义Rastrigin函数（经典非凸测试函数）
- 从相同初始点运行不同优化器
- 绑制收敛轨迹

**数据来源**：
- Rastrigin函数：f(x) = 10n + Σ(x²-10cos(2πx))
- 初始点随机选取

---

## algorithm_comparison.pdf

**背景**：总结各类优化算法的适用场景。

**意图**：给读者一个选择优化算法的决策参考。

**生成方法**：
- 代码文件：`code_chap02/optimization_tools_demo.py`
- 汇总不同问题类型的求解时间
- 绘制对比柱状图

**数据来源**：
- LP、QP、非凸优化的综合测试结果
