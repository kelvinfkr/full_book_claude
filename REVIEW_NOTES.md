# 全书审读记录：问题清单与本次修订

本文件记录一次对《大二开始的物理智能》全书的审读。目标是回答两个问题：

1. 后面各章在**写作逻辑**上与第 1–3 章差在哪里？
2. 全书在**内容逻辑**上有哪些前后不一致、跨章引用错误、推导或数值错误？

第 1–3 章的写作逻辑可以概括为一个固定模板，后面各章已按这个模板逐章对齐（见"本次修订"）。仍需作者决定的事项列在"待作者决定"。

## 一、第 1–3 章的写作逻辑（作为全书基准）

| 环节 | 第 1–3 章的做法 |
|---|---|
| 开篇 | `\section*{}` 无编号引言：一位历史人物的故事（`\storynote`）→ 一个"常见误解 / 反直觉现象" → 一段斜体的日常反问 → `本章目标` 一段话 → "学完这章，你将能够…" |
| 第一节 | 从一个具体的小问题出发，先展示朴素方法及其缺陷，再引出新思想 |
| 核心观点 | 一个黄色盒子明说"本章的问题就是第 1 章的拉格朗日乘子法，只是……"，并给出"第 1 章 vs 本章"的对照表（变量 / 目标 / 约束 / 乘子 / 一阶条件 / 影子价格） |
| 主线 | 每章都回答"约束是什么？$\lambda$ 代表什么？"，并用 `physicalmeaning` 盒子讲 $\lambda$ 的物理 / 经济意义 |
| 页边栏 | `\preview`（预告）、`\review`（回顾）、`\think`（想一想）、`\clarify`（澄清）、`\storynote`（人物）、`\tryit`（动手试试） |
| 结尾 | `本章小结`（编号列表）→ "$\lambda$ 在本章的意义"盒子（前后章呼应）→ `习题`（计算题 / 证明题 / 编程题 / 思考题，编号连续）→ `\section*{扩展阅读}` |

审读前的统计（第 10–16 章几乎没有页边栏、没有核心观点盒子、没有 $\lambda$ 盒子，习题分类各自为政）：

| 章 | 本章目标 | 核心观点盒 | 页边栏（preview/review/think/clarify/tryit） | λ 盒子 | 习题分类 |
|---|---|---|---|---|---|
| 1–3 | 有 | 有 | 每章 5–10 条 | 有 | 计算/证明/编程/思考 |
| 4–9 | 部分有 | 无 | 0–4 条 | 部分有 | 基本一致 |
| 10–16 | 列表式或无 | 无 | 0–2 条 | 无 | 各章自定名称、`exercise` 盒子、无编号 |

## 二、审读中提出的问题

### A. 全书结构与主线

1. 附录"章节速查"表和"跨章对照表"是旧版大纲（"线性代数回顾 / 矩阵求导 / SVM / Ch12-5"），与实际章节完全对不上；`book_outline.tex` 里第 11、12 章的注释也标错了（"图神经网络 / 序列建模"）。
2. 附录内容在 `book_outline.tex` 和 `chapters/appendix_index.tex` 里各有一份，后者从未被 `\input`。
3. 第 1 章末尾预告"第 10 章：$\lambda$ = 值函数的梯度"，第 10 章开篇却说"值函数 $V(s)$ 就是乘子"，第 10 章正文推导的又是 $\lambda_t = \gamma^{t+1}\nabla V$——三处口径不一。
4. 第 8 章连续部分把约束写成 $\lambda^\top(\dot x - f)$，却给出 $\dot\lambda = -\partial H/\partial x$，二者不相容；结果连续部分 $\lambda = +\partial J^*/\partial x$，离散部分 $\lambda_N = -\partial\phi/\partial x_N$，与第 1、9 章符号相反，全书没有任何一处说明。
5. 第 11 章伴随方程的符号与其约束写法相反；特征值、K–L 系数都用 $\lambda$ 记号，与"全书主角 $\lambda$ 是乘子"冲突。
6. 第 13 章把损失权重、拉格朗日乘子、对偶步长三个东西都记作 $\lambda$ 或 $\rho$，与第 1 章"$\rho$ 是罚系数、$\lambda$ 是乘子"的约定打架；等式约束记号 $h(x)=0$ 与第 1 章的 $g(x)=0$ 对调。
7. 第 14 章通篇没有出现"拉格朗日 / 乘子 / 影子价格"任何一个词，是唯一完全脱离主线的一章。
8. 第 12 章（工程章）如何纳入主线？——本次采用"显存 / 算力 / 带宽是硬约束，它们的乘子是资源的影子价格"这一表述。
9. 第 16 章缺 `本章小结`，习题直接接在正文后；`figs_chap16/unified_view.png` 从未被引用。
10. 第 7 章开篇宣称"关节力矩也是拉格朗日乘子"，正文里 $\tau$ 始终是拉格朗日方程右端的广义力，从未作为乘子出现；第 1 章两次预告"第 7 章会出现 KKT 分块矩阵（质量矩阵 + 约束雅可比）"，第 7 章没有写出来。

### B. 跨章引用错误（已全部修正）

- 第 4 章四处"第 4 章的 E-L 方程"（自引，应为第 3 章）；扩展阅读"第 4 章的约束力"应为"本章"。
- 第 5 章"第 4 章的 Euler-Lagrange 方程"、"第 5 章力学中的约束力"（自引）、"第 1 章中消去部分变量得到约束力"。
- 第 10 章"第 2–3 章花大量篇幅讲线性代数和矩阵求导"（第 2 章是对偶、第 3 章是变分法）；扩展阅读"在第一章中我们推导出了贝尔曼方程"（应为本章）；"从第 3 章的梯度计算"。
- 第 16 章两处"第一章……拉格朗日力学"（应为第 4 章）。
- 第 7 章三处"线性规划的互补松弛"未注明第 2 章。

### C. 重复与结构错误（已修正）

- 第 8 章两个一模一样的 `\section{本章小结}`；第 10 章两个同名 `\section{用拉格朗日乘子法推导贝尔曼方程}`。
- 第 9 章一个 110 行的小节（"Step 5：完整示例"）整段重复；优化器一节把"训练就是 $\min_\theta L(\theta)$"讲了三遍。
- 第 5 章 FSI 扩展阅读末尾整块重复（两张图、两个 `\label`、两个参考文献块、一段引用不存在模型的"对比"）。
- 第 16 章文件被破坏：每行之后插入了空行、原空行变成只含一个空格的行（已还原）。
- 第 5 章 GIL / 向量化实验图放在正文小结前，实际属于 CPU 并行扩展阅读；第 9 章图注描述的函数与图中不符。

### D. 推导、数值与史实错误（已修正）

- 第 4 章"为什么是 $T-V$"的论证（"$T+V$ 守恒所以无法区分路径"）是经典误论；单摆两个例子坐标约定自相矛盾；"$\lambda$ 就是张力"与该章自己算出的 $T = 2|\lambda| l$ 矛盾。
- 第 5 章热流守恒推导符号写反（得到 $-u'' = -f$）；"凹 / 凸"术语写反；热传导泛函表中 $\pm fu$ 符号错。
- 第 7 章牛顿迭代手算数值错误（书中 $\Delta q$ 代回 $J$ 得不到误差向量），已换初值重算；奇异位形下"伪逆"说法错误；零空间次级目标符号写反；"力放大"说反；综合案例质量模型前后矛盾。
- 第 8 章 MPC 例题多出一项 "+20"，最优解 $(-7.34, 5.29)$ 应为 $(-6.45, 4.40)$，下游数值连锁更新；一个只写了一半就放弃的 $N=2$ 例子。
- 第 10 章绳网例子里 $b \to c$ 边其实也绷直（对偶解不唯一），正文断言错误；对偶上升算法的初始化和更新公式与自己的约定矛盾。
- 第 11 章数据处理不等式链方向写反；KV Cache 算例应为 1 GB 而非 2 GB。
- 第 12 章 FLOPs 公式把 $S$ 定义为序列长度却标"每个 token"；64×H100 预训练时间表与自身算例矛盾（42 / 78 天、1.5 年应为 15 / 28 天、7 个月）；KV Cache 速查表整体偏小一半；代码缺 `import torch.nn as nn`。
- 第 13 章图注与正文对梯度范数的描述相反（图中 PDE 梯度更大）；因果权重公式依赖未积分的 $x$；傅里叶特征经验法则与实验矛盾；"Deep Ritz 对应最小作用量原理"应为最小势能原理。
- 第 14 章离散化无关性的极限断言过强；统一表中 DeepONet 的核写错；格林函数定理漏边界条件；Galerkin 注意力描述与原文不符。
- 第 15 章 QMIX 的 IGM 证明只检查了单方偏离；HJB 缺 $-\nu\Delta u$ 项；MFG 例子终端代价 $\tfrac12 x^2$ 与图注 $x^2$ 矛盾；"拥挤代价越远越好"符号写反；MFG 图并非数值解（代码手绘），已改称示意；Lasry–Lions "独立提出"史实错。
- 第 16 章 CBF 闭式解、伪代码、代码三处符号写反（修正沿 $-\nabla h$ 推向障碍物），配套图中 CBF 从未激活。已修正文字、伪代码和 `code_chap16` / `code_organized` 两份代码。
- 史实：牛顿生年（儒略历）、Linnainmaa 是硕士论文、Kingma 当时在阿姆斯特丹、Sutton 已获 2024 年图灵奖、纳什 21 岁证明存在性等。

### E. 承诺未兑现（已处理）

- 第 5 章"用 Python 组装刚度矩阵并求解"：正文无代码 → 新增"十行代码：一维线性有限元"。
- 第 8 章"用直接法数值求解"、"机器人案例"：→ 新增编程题（直接法、MPC），并明确"直接法"一词。
- 第 7 章"用 D-H 参数建模"：正文只有页边栏一句 → 学习目标改为"齐次变换（D-H 是它的标准化写法）"。
- 第 9 章"手推 3 层网络 / PyTorch 分类器"：正文是两层、无 PyTorch 代码 → 学习目标改写。
- 第 11 章"手算 Attention / PyTorch 实现 Transformer 块"：→ 新增两道编程题与 `\tryit`。
- 第 12 章"梯度检查点"：只有一句带过 → 新增澄清页边栏；"本章分四部分"漏掉 LoRA、FlashAttention/MoE → 改为六部分。
- 第 13 章"NTK 权重 / 对比传统方法"：正文没有 → 学习目标改写。
- 第 16 章"CBF、CLF"、"可微物理的局限"：正文没有 CLF、没有局限 → 学习目标改写，新增"局限"一段。

## 三、本次修订做了什么

1. **全书统一模板**：第 4–16 章每章补齐（或改写为第 1 章口吻）：斜体反问 + `本章目标` 一段话；"本章的核心观点"黄色盒子 + "第 1 章 vs 本章"对照表；"$\lambda$ 在本章的意义"盒子并与前后章呼应；`本章小结` 编号列表；习题统一为计算题 / 证明题 / 编程题 / 思考题并连续编号；扩展阅读统一为 `\section*` + `\addcontentsline`。
2. **页边栏**：第 4–16 章共新增约 120 条 `\preview / \review / \think / \clarify / \tryit / \storynote`，人物小传统一改用 `\storynote`。
3. **符号约定**：第 8、9、10、11 章分别加入"符号约定提醒"，明确各章 $\lambda$ 的正负号来源；第 11 章特征值改记 $\mu$；第 13 章明确"权重 $\lambda$ 扮演的是 $\rho$ 的角色"，对偶步长改记 $\eta_\lambda$。
4. **B–E 各项**：全部按上文修正。
5. **附录**：重写 `chapters/appendix_index.tex`（新增"各章的目标 / 约束 / $\lambda$"总表，修正跨章对照表、快速定义、概念图谱、章节速查），`book_outline.tex` 改为 `\input` 该文件，修正章节注释。
6. **LaTeX 用法**：`\begin{keyidea}{X}` / `[title=X]` 统一为 `[X]`（该环境把参数直接作标题）；`\begin{exercise}[中文标题]`、`\begin{advantage}[…]` 改为合法写法；`\begin{marginnote}…\end{marginnote}`（无此环境）改为 `\marginnote{…}`；数学模式中的 `°` 改为 `^\circ`；第 13 章图注引用改为 `\eqref`。

## 四、第二轮修订：原"待作者决定"各项的处理结果

作者回复"都改了吧，图不合适的重新生成一批，像教科书一点，每张图都有思路"，于是第一轮列出的 8 项全部处理：

1. **第 8 章离散部分与第 9 章翻号，全书动态章节统一符号**。动力学约束一律写成 $f - \dot x$（离散版 $f(x_k,u_k) - x_{k+1}$，第 9 章 $f_i(z_{i-1}) - z_i$），于是所有动态章节都有 $\lambda = +\partial(\text{代价})/\partial(\text{状态})$：第 8 章连续/离散、第 9 章 $\lambda_i = \partial L/\partial z_i = \delta_i$（正好是深度学习的误差信号，不再"差一个负号"）、第 10 章 $\lambda_t = \gamma^{t+1}\nabla V$、第 11 章 BPTT。第 8 章离散推导改用离散哈密顿量 $H_k = L_k + \lambda_{k+1}^\top f$，伴随方程 $\lambda_k = \partial H_k/\partial x_k$、$\lambda_N = \partial\phi/\partial x_N$，与连续部分逐项对应；手算例题（$N=2$ 积分器）重算，最优控制 $u_0=u_1=-10/21$ 不变。第 1 章的 $\lambda = -df^*/d\epsilon$ 规则在各章的"符号约定"页边栏里重新解释（放松 $f - x = \epsilon$ 等于把状态往回拨 $\epsilon$，两个负号抵消）。第 9 章的对照表、反向传播伪代码、两层网络手算例、扩展阅读中的拉格朗日函数同步翻号。
2. **第 13 章损失权重改记 $\rho$**：正文（开篇反问、罚参数讨论、梯度竞争公式、实验表格、对比表、逆问题损失）与代码清单里的固定权重全部由 $\lambda$ 改为 $\rho$，`lambda_bc` → `rho_bc`（`code_chap13/`、`code_organized/` 中 5 个固定权重脚本与 README 同步）；对偶上升里真正的乘子仍记 $\lambda$（`self.lambda_bc`）。图 `figs_chap13/lambda_effect.pdf` 用统一风格重跑（$\rho \in \{0.1,1,10,100,1000\}$，10000 轮），文件名保留以免改引用，正文表格按新结果更新。
3. **第 12 章 1500 行"服务器与环境配置"移为附录 B**（`chapters/appendix_env.tex`，标题"实验环境速成——服务器与环境配置"），原位置留一段"关于实验环境"指引；原附录改称"附录 A：核心概念索引"；`book_outline.tex` 增加 `\input`。
4. **第 10 章扩展阅读**：五篇之前加"扩展阅读导读"盒子（每篇一句话说明内容、前置知识、适合谁读），四篇各自补"本节导读"（第四篇原有）。未合并正文，避免破坏各篇的自洽。
5. **图片改名**：`figs_chap11/chap12_fig*.png` → `chap11_fig*.png`，标签 `fig:chap12_*` → `fig:chap11_*`，并重新生成（见第五节）。
6. **`[h]` → `[htbp]`**：全书 `\begin{figure}[h]` / `\begin{table}[h]` 统一改为 `[htbp]`。
7. 第 16 章正文中"请重新运行脚本再生成"的 TODO 注释删除（图已重绘）。
8. 引号与 `\hline`/booktabs 仍未逐处统一（排版层面，与本轮目标无关）。

## 五、图片重绘与统一风格

### 5.1 审图结论

逐张看过全书 70 张被引用的图（PDF 用 PyMuPDF 渲染后查看）。保留不动的：第 1、3 章全部；第 4 章 `chap04_fig1`；第 5 章 `chap05_fig1/3` 与 Turek–Hron 四张（来自真实 Kratos 仿真）；第 8 章两张；第 13 章除 `lambda_effect` 外全部；第 14 章；第 15 章除 `mfg_solution` 外全部；第 16 章 `phase_space`、`free_energy`、`paradigm_comparison`。需要重绘的问题分四类：

- **内容错误或"假数据"**：第 4 章 `chap04_fig2` 的周期误差面板算错（用首次过零时刻×2，实际测的是半周期，得出 −50%）；第 5 章 `chap05_fig2` 的"节点误差"是 $10^{-15}$（线性元节点超收敛），$O(h^2)$ 参考线没有意义；第 10 章 `cartpole_demo` 右图的训练曲线是随机数画的（标题写着 Simulated）；第 15 章 `mfg_solution` 是解析高斯与抛物线手绘的"示意解"；第 16 章 `cbf_controller`、`multi_obstacle` 由符号写反的旧代码生成（CBF 从未激活）；第 11 章注意力图的权重几乎均匀（没有信息量）。
- **与正文不符**：第 10 章正文五处写 GridWorld 是 $5\times5$，图全是 $4\times4$；第 7 章 IK 图的起点与正文算例 $q^{(0)}=(0^\circ,90^\circ)$ 不一致；第 4 章 `chap04_fig1` 图注"约 6 秒"与图中 11.4 s 不符。
- **不是图**：第 9 章 `chap09_fig1` 第 4 面板是一段文字表格；第 11 章两张图里有大段文字框。
- **风格**：第 2 章 4 张、第 9 章 5 张、第 10 章扩展阅读 4 张全是英文标签；各章字号、配色、标题风格各异；第 4 章 `long_time_comparison` 的 Verlet 振荡填满整幅、RK4 漂移看不见；第 10 章 `chap10_shortest_path` 文字重叠；`chap04_fig2` 有方框字。

### 5.2 统一风格：`code/textbook_style.py`

新增全书共用的绘图风格模块（所有重绘脚本都 `from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS`）：

1. 图内不放大标题，用 **(a)(b)(c)** 面板编号，结论写在 caption 里；
2. 坐标轴、图例、注记全部中文，公式用 mathtext；字体优先 **Noto Sans CJK SC**（同时含中文、西文和数学负号 U+2212，解决了文泉驿字体下对数坐标刻度的负号变方框的问题），缺失时回退到 DejaVu Sans + 文泉驿；
3. 固定 8 色语义色板：精确解/参考黑色，数值解/学习结果蓝色，误差/警示/最优路径红色，约束/可行域/安全集绿色；
4. 单栏宽 6.3 in、按面板数自动给高度，线宽 1.6、字号 9–10 pt，去上/右轴线，浅灰网格；
5. `save_figure` 同时输出 PDF（矢量，供 LaTeX）与 300 dpi PNG，正文引用的文件名不变。

### 5.3 重绘清单（按章）

每张图在 `figure_docs/chapNN_figures.md` 里都有"背景 / 意图 / 生成方法 / 数据来源 / **思路** / **读图指南**"六段，这里只列一句话思路。

| 章 | 图 | 脚本 | 一句话思路 |
|---|---|---|---|
| 2 | `lp_performance`、`qp_performance`、`portfolio_weights`、`nonconvex_comparison` | `code_chap02/optimization_tools_demo.py` | 求解逻辑保留、全部中文化：LP 是最便宜的问题（1000 变量 0.3 s）；通用 QP 求解器代价 $\propto n^{3.2}$；最优组合只集中在 4 种资产上（收益约束不起作用，是预算与非负约束在塑造解）；Rosenbrock 上有梯度方法最省。LP 代码清单原来的"输出"是假的（$c\ge0$ 求最小，解应为 0），改为最大化利润并换成真实输出 $x=(2.2,1.8,1.6)$、利润 18.2。 |
| 4 | `chap04_fig2` | `code/chap04_mechanics.py` | 线性化不是对错而是"误差多大"：过零事件实测周期偏差（30°→+1.7%，90°→+18.0%，与椭圆积分公式一致）；`chap04_fig1` 图注改为"约 11 秒"。 |
| 4 | `long_time_comparison`、`harmonic_comparison`、`convergence_order` | `code_chap04/ode_numerical_methods.py` | RK4 能量误差每周期最大值在双对数下是斜率 0.96 的直线（无界），Verlet 是水平线（有界振荡，无漂移）；收敛阶图例直接给出拟合斜率。 |
| 5 | `chap05_fig2` | `code/chap05_performance.py fem` | 有限元的误差要说清是哪种范数：节点精确（超收敛）、$L^2$ 与单元内最大误差 $O(h^2)$、$H^1$ 半范数 $O(h)$，并补上 caption 提到的三对角刚度矩阵面板。 |
| 7 | `chap07_fig1` | `code/chap07_robotics.py` | 正文手算例（$L_1=L_2=1$、$x_d=(1.2,0.9)$、$q^{(0)}=(0^\circ,90^\circ)$）真正迭代：误差 $0.22\to0.021\to2\times10^{-4}\to2\times10^{-8}$ 的二次收敛，$|\det J|$ 热图解释为什么目标点远离奇异位形；图移到算例之后。 |
| 9 | `activation_functions`、`gradient_flow`、`computation_graph`、`gradient_check`、`modern_activations` | `code_chap09/activation_functions.py` | 全部中文化；`gradient_flow` 的"梯度消失"与"死亡 ReLU"改为真实实验（20 层网络各层梯度范数比；3 层 MLP 训练中恒为 0 的单元比例，ReLU 大学习率 10%→57%，Leaky ReLU 可恢复）。 |
| 9 | `chap09_fig1` | `code/chap09_autograd.py` | 第 4 面板改为真图：$N$ 层残差网络反向传播得到的 $\lambda_i=\delta_i$ 落在连续协态 $p(t)$ 上，偏差随 $h=T/N$ 一阶下降——反向传播就是伴随方程的离散形式。 |
| 10 | `chap10_shortest_path`、`chap10_value_iteration`、`chap10_value_heatmap`、`chap10_policy`、`chap10_qlearning_sarsa`、`chap10_qtable` | `code_chap10/chap10_rl.py` | 环境改为正文的 $5\times5$（每步 $-1$、$\gamma=1$）：离终点 $d$ 步的状态恰在第 $d$ 轮到位；$V$ = 负曼哈顿距离；并列最优动作画双箭头；Q-Learning/SARSA 10 个种子真实训练；Q 表与值迭代逐格相同。 |
| 10 | `arm_kinematics`、`rl_control_comparison`、`pd_control_trajectory`、`cartpole_demo` | `code_chap10/robotic_arm_control.py` | 中文化；CartPole 自行实现（Barto 方程，不依赖 gym），用交叉熵法 + 线性策略（600 回合达到 195）和表格 Q-Learning 得到真实学习曲线，替换原来的随机数曲线。 |
| 11 | `chap11_fig1`、`chap11_fig2`（原 `chap12_fig*`） | `code_chap11/chap11_figures.py` | 在含噪正弦上训练单头自注意力，注意力矩阵出现间距 16 的斜条纹（"关注同相位时刻"）；正弦位置编码矩阵、各维波长、位置间点积的 Toeplitz 结构。 |
| 13 | `lambda_effect` | `code_chap13/lambda_effect.py` | $\rho\in\{0.1,\dots,1000\}\times3$ 个种子各 10000 轮：$\rho$ 太小前期慢（第 1000 轮误差高 3–4 个量级、解整体漂一个齐次解），$\rho=10\sim100$ 前期最快，$\rho$ 太大后期震荡、最终反而最差——"罚得越狠边界学得越差"，为自适应权重铺垫。正文表格与结论按真实结果重写（原表格数字不可复现）。 |
| 15 | `mfg_solution` | `code_chap15/chap15_mfg_solver.py` | 真正的一维 MFG 有限差分求解：HJB 倒向（隐式扩散 + 迎风 Hamilton 项）、FP 正向（守恒迎风格式）、阻尼不动点迭代 112 轮残差降到 $10^{-8}$；展示"个体按 HJB 最优响应、群体按 FP 演化、均衡是不动点"，拥挤项 $\kappa m$ 就是"别人对我的价格"。 |
| 16 | `cbf_controller`、`multi_obstacle`、`differentiable_physics`、`latent_dynamics`、`unified_view` | `code_chap16/chap16_figures_v2.py` | CBF 用正确符号 $u^*=u_{\rm ref}+\lambda^*\nabla h$ 重做，障碍物移出起点–目标连线，画出 $h(t)\ge0$ 与只在贴边时非零的 $\lambda^*(t)$（安全约束的影子价格）；多障碍版显示两条约束同时活跃；可微物理用 autograd 穿过 200 步 RK4 真正辨识出 $(m,k,c)$；隐空间动力学用 $24\times24$ 带噪单摆图像训练编码器 + 转移模型，潜态是嵌套闭合环、30 步"做梦"误差贴近噪声；统一视角图改为全中文、内容取自附录 A 主线总表。 |

### 5.4 遗留

- 仍是旧风格但内容无误、未重绘的图：第 13 章其余 12 张（中文、清晰）、第 15 章其余 6 张、第 16 章 3 张、第 5 章 Turek–Hron 四张、第 4 章 `chap04_fig1`、第 14 章 `chap14_fig1`、第 1–3 章。
- 旧的重复脚本未删（`code/chap10_rl.py`、`code/chap12_transformer.py`、`code_chap07/chap07_robotics.py`、`code_chap12/chap12_transformer.py`、`code_chap16/chap16_world_model.py`），它们生成的是旧图或写到旧路径，建议清理；新图一律由 `figure_docs` 中标明的脚本生成。
- 本环境没有 TeX，无法编译；对所有改动过的章节做了 `\begin/\end` 与花括号配对检查。
- 计时类图（第 2 章）的绝对秒数依赖机器，caption 用了"约"。

## 六、编译验证（TeX Live 2023，XeLaTeX）

在容器里装了 TeX Live（`texlive-xetex`、`texlive-lang-chinese`、`texlive-latex-extra`、`texlive-science`、`texlive-pictures`、`texlive-fonts-extra`），用 `xelatex -output-directory=build book_outline.tex` 连跑多遍，最终 **0 个错误、0 个未定义引用，664 页**。为此做的修正：

1. **仓库根目录下的 27 个 `.sty` 存根**（`tcolorbox.sty`、`hyperref.sty`、`listings.sty`……，早先没有 TeX 时写的"兼容实现"）会屏蔽 TeX Live 的真包，导致上百个错误；已全部移到 `latex_stubs/`（保留备查，编译时不再被找到）。
2. `\newfontfamily\cyrillicfont{Times New Roman}` 在没有该字体的机器上直接报错，改为 `\IfFontExistsTF` 回退到 DejaVu Serif；第 6 章的生僻字"龘、靐"Fandol 字体没有，加了 `\rarecjkfont`（有 Noto Serif CJK 时用它兜底）。
3. 第 5 章：`lstlisting` 的 caption 含 `=` 和 `$`，加花括号；`\begin{references}` 不存在，改为 `thebibliography`。
4. `book_outline.tex` 结尾的 `\textit{...}` 内有空行，导致"Paragraph ended"错误；已合并。
5. 缺字：文本模式的 ✓/✗/≈/∝ 改为 `\checkmark`、`$\times$`、`$\approx$`、`$\propto$`；数学下标里的中文与全角冒号（`d_{可行}`、`_{L_{\text{PDE}}：...}`）包进 `\text{}`；代码清单里的 α、λ、∇、²、· 通过 `literate` 映射为数学符号。
6. `.gitignore` 增加 `build/`。

剩余警告只有 hyperref 的"Token not allowed in a PDF string"（章节标题含公式，影响书签文字，不影响正文）和 fontspec 对 Fandol 字体缺少某些 OpenType 特性的提示，均无害。
