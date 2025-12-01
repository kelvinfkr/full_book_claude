# 第15章 图片说明

## prisoners_dilemma.pdf / prisoners_dilemma.png

**背景**：第15章介绍多智能体系统与博弈论。

**意图**：用经典的囚徒困境说明博弈论基本概念。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 绘制支付矩阵
- 标注纳什均衡

**数据来源**：
- 经典囚徒困境支付矩阵
- (合作,合作)=(-1,-1)
- (背叛,背叛)=(-3,-3)
- 纳什均衡：(背叛,背叛)

---

## potential_game.pdf / potential_game.png

**背景**：势博弈是一类特殊的多人博弈。

**意图**：说明势函数如何简化均衡分析。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 绘制势函数等高线
- 标注各玩家的最优响应

**数据来源**：
- 拥堵博弈示例
- 势函数：Φ(a) = Σ cost(a)

---

## network_game.pdf / network_game.png

**背景**：网络博弈发生在图结构上。

**意图**：展示网络结构如何影响博弈均衡。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- networkx绘制博弈网络
- 节点颜色表示策略

**数据来源**：
- 20节点随机网络
- 协调博弈设置
- 均衡状态可视化

---

## braess_paradox.pdf / braess_paradox.png

**背景**：Braess悖论是交通网络的经典现象。

**意图**：说明增加道路反而可能增加拥堵。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 绘制交通网络拓扑
- 对比有无新路的均衡流量

**数据来源**：
- 经典Braess网络（4节点）
- 用户均衡计算
- 总行程时间对比

---

## mfg_solution.pdf / mfg_solution.png

**背景**：平均场博弈处理大规模多智能体系统。

**意图**：展示平均场均衡的密度分布演化。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 求解HJB-FP耦合方程
- 绘制密度m(x,t)的时空演化

**数据来源**：
- 线性二次平均场博弈
- 有限差分数值解
- 时间T=1，空间x∈[0,1]

---

## ctde_comparison.pdf / ctde_comparison.png

**背景**：集中训练分散执行（CTDE）是多智能体RL的主流范式。

**意图**：对比不同MARL架构：独立学习、完全集中、CTDE。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 架构示意图
- 性能曲线对比

**数据来源**：
- 概念性架构图
- 合作导航任务的学习曲线

---

## qmix_architecture.pdf / qmix_architecture.png

**背景**：QMIX是著名的值分解方法。

**意图**：图解QMIX如何将个体Q值混合成全局Q值。

**生成方法**：
- 代码文件：`code_chap15/chap15_generate_figures.py`
- 网络架构图绘制
- 标注单调性约束

**数据来源**：
- QMIX论文架构
- 混合网络权重由超网络生成
