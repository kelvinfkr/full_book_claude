# 第11章 图片说明

## chap12_fig1.png

**背景**：第11章介绍从RNN到Transformer的序列建模演进。

**意图**：可视化RNN的梯度消失问题——为什么长序列难以训练。

**生成方法**：
- 代码文件：`code_chap12/chap12_transformer.py`
- 构建多层RNN
- 计算BPTT梯度并可视化衰减

**数据来源**：
- 序列长度：100
- 隐藏层大小：64
- 统计各时间步的梯度范数

---

## chap12_fig2.png

**背景**：Self-Attention是Transformer的核心机制。

**意图**：可视化注意力权重矩阵，展示"谁在关注谁"。

**生成方法**：
- 代码文件：`code_chap12/chap12_transformer.py`
- 实现scaled dot-product attention
- 热图绘制注意力矩阵

**数据来源**：
- 示例句子的token序列
- 注意力计算：softmax(QK^T/√d)
