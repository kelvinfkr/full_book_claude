# 第10章 图片说明

## arm_kinematics.pdf

**背景**：第10章介绍强化学习在控制中的应用。

**意图**：展示机械臂控制任务的设置。

**生成方法**：
- 代码文件：`code_chap10/robotic_arm_control.py`
- 绘制机械臂和目标点
- 标注状态空间和动作空间

**数据来源**：
- 2自由度平面机械臂
- 状态：[θ1, θ2, ω1, ω2]
- 动作：[τ1, τ2]（关节力矩）

---

## pd_control_trajectory.pdf

**背景**：PD控制是经典的反馈控制方法。

**意图**：展示PD控制器的轨迹跟踪效果，作为RL的基准对比。

**生成方法**：
- 代码文件：`code_chap10/robotic_arm_control.py`
- 实现PD控制器
- 绘制跟踪误差曲线

**数据来源**：
- PD增益：Kp=100, Kd=10
- 目标轨迹：圆形
- 采样频率：100Hz

---

## rl_control_comparison.pdf

**背景**：对比RL与经典控制。

**意图**：展示PPO/SAC学习到的控制策略与PD控制的对比。

**生成方法**：
- 代码文件：`code_chap10/robotic_arm_control.py`
- 训练RL agent
- 对比轨迹和能量消耗

**数据来源**：
- 训练回合：1000
- 奖励函数：-||位置误差|| - 0.01||力矩||

---

## cartpole_demo.pdf

**背景**：CartPole是RL的经典入门环境。

**意图**：可视化倒立摆的平衡控制问题。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 使用gym的CartPole环境
- 绘制状态序列快照

**数据来源**：
- OpenAI Gym CartPole-v1
- 状态：[x, v, θ, ω]
- 动作：左/右推力

---

## chap10_value_iteration.png

**背景**：值迭代是动态规划的基础算法。

**意图**：展示值函数如何逐步收敛。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 网格世界环境
- 绘制每次迭代后的V(s)

**数据来源**：
- 5×5网格世界
- 折扣因子γ=0.9
- 收敛阈值：1e-4

---

## chap10_value_heatmap.png

**背景**：可视化最终的值函数。

**意图**：用热图展示状态价值的空间分布。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 值迭代收敛后的V(s)
- matplotlib热图绘制

**数据来源**：
- 收敛后的值函数
- 颜色映射：高价值=暖色

---

## chap10_qtable.png

**背景**：Q表是表格型RL的核心数据结构。

**意图**：展示Q(s,a)的学习结果。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- Q-learning训练后的Q表
- 热图+箭头表示最优动作

**数据来源**：
- 训练episode：500
- 学习率α=0.1
- ε-greedy探索

---

## chap10_qlearning_sarsa.png

**背景**：对比off-policy和on-policy方法。

**意图**：展示Q-learning与SARSA在cliff walking问题上的差异。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 悬崖行走环境
- 对比学到的策略路径

**数据来源**：
- Q-learning：最优但冒险
- SARSA：保守但安全

---

## chap10_shortest_path.png

**背景**：最短路径是RL的基础应用。

**意图**：展示RL学到的从起点到终点的路径。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 网格世界中的路径规划
- 绘制学到的轨迹

**数据来源**：
- 网格世界带障碍
- 起点：左上角
- 终点：右下角

---

## chap10_policy.png

**背景**：策略可视化。

**意图**：用箭头展示每个状态的最优动作。

**生成方法**：
- 代码文件：`code_chap10/chap10_rl.py`
- 从Q表提取贪婪策略
- 箭头图绘制

**数据来源**：
- π(s) = argmax_a Q(s,a)
- 四个方向：上下左右
