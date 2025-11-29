"""
第12章：从RNN到Transformer - 可视化演示

包含：
1. Attention权重可视化
2. 位置编码演示
3. 自注意力计算示例
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# 导入中文字体配置
import sys
sys.path.insert(0, '/home/user/full_book/code')
from plot_utils import setup_chinese_font
setup_chinese_font()


def softmax(x, axis=-1):
    """数值稳定的softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def create_attention_visualization():
    """
    自注意力机制的完整计算过程可视化
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 模拟3个词的嵌入 (seq_len=3, d_model=4)
    np.random.seed(42)

    # 词嵌入矩阵 X (3x4)
    words = ['我', '爱', 'NLP']
    X = np.array([
        [1.0, 0.0, 0.5, 0.2],   # 我
        [0.5, 1.0, 0.3, 0.8],   # 爱
        [0.2, 0.5, 1.0, 0.1]    # NLP
    ])

    # 权重矩阵 (4x4)
    d_k = 4
    W_Q = np.random.randn(4, d_k) * 0.3
    W_K = np.random.randn(4, d_k) * 0.3
    W_V = np.random.randn(4, d_k) * 0.3

    # 计算Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 计算注意力分数
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)

    # 计算输出
    output = attention_weights @ V

    # === 子图1: 输入嵌入 ===
    ax1 = axes[0, 0]
    im1 = ax1.imshow(X, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f'd{i}' for i in range(4)])
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(words)
    ax1.set_title('输入嵌入矩阵 X (3×4)', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    # 在格子里标注数值
    for i in range(3):
        for j in range(4):
            ax1.text(j, i, f'{X[i,j]:.2f}', ha='center', va='center', fontsize=9)

    # === 子图2: Q, K, V矩阵 ===
    ax2 = axes[0, 1]

    # 并排显示Q, K, V
    combined = np.hstack([Q, np.ones((3, 1))*np.nan, K, np.ones((3, 1))*np.nan, V])
    im2 = ax2.imshow(combined, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.axvline(x=3.5, color='black', linewidth=2)
    ax2.axvline(x=8.5, color='black', linewidth=2)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(words)
    ax2.set_xticks([1.5, 6.5, 11.5])
    ax2.set_xticklabels(['Q = XW_Q', 'K = XW_K', 'V = XW_V'], fontsize=10)
    ax2.set_title('查询、键、值矩阵', fontsize=12)

    # === 子图3: 注意力权重热图 ===
    ax3 = axes[0, 2]
    im3 = ax3.imshow(attention_weights, cmap='Oranges', vmin=0, vmax=1)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(words)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(words)
    ax3.set_xlabel('键 (Key)', fontsize=11, labelpad=10)
    ax3.set_ylabel('查询 (Query)', fontsize=11, labelpad=10)
    ax3.set_title('注意力权重: softmax(QK^T/√d_k)', fontsize=12)
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    # 在格子里标注数值
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{attention_weights[i,j]:.2f}', ha='center', va='center',
                     fontsize=10, color='black' if attention_weights[i,j] < 0.5 else 'white')

    # === 子图4: 计算流程图 ===
    ax4 = axes[1, 0]
    ax4.axis('off')

    # 绘制计算流程
    steps = [
        '步骤1: Q = X × W_Q\n(查询向量：我要找什么)',
        '步骤2: K = X × W_K\n(键向量：我有什么信息)',
        '步骤3: scores = Q × K^T / √d_k\n(计算相似度分数)',
        '步骤4: weights = softmax(scores)\n(归一化为概率分布)',
        '步骤5: output = weights × V\n(加权求和)'
    ]

    for i, step in enumerate(steps):
        ax4.text(0.1, 0.85 - i*0.18, step, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                 transform=ax4.transAxes, verticalalignment='top')
        if i < len(steps) - 1:
            ax4.annotate('', xy=(0.5, 0.78 - i*0.18), xytext=(0.5, 0.72 - i*0.18),
                         arrowprops=dict(arrowstyle='->', color='gray'),
                         transform=ax4.transAxes)

    ax4.set_title('自注意力计算流程', fontsize=12)

    # === 子图5: 单词"爱"的注意力分布 ===
    ax5 = axes[1, 1]

    word_idx = 1  # "爱"
    weights_love = attention_weights[word_idx]

    bars = ax5.bar(words, weights_love, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax5.set_ylabel('注意力权重', fontsize=11, labelpad=10)
    ax5.set_title(f'"{words[word_idx]}"关注哪些词?', fontsize=12)
    ax5.set_ylim(0, 1)

    for bar, w in zip(bars, weights_love):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{w:.2f}', ha='center', fontsize=11, fontweight='bold')

    # === 子图6: 多头注意力示意 ===
    ax6 = axes[1, 2]
    ax6.axis('off')

    # 绘制多头注意力的示意图
    n_heads = 4
    colors = plt.cm.Set3(np.linspace(0, 1, n_heads))

    for i in range(n_heads):
        rect_x = 0.1 + i * 0.22
        rect_width = 0.18
        rect_height = 0.3

        # 绘制每个head
        rect = plt.Rectangle((rect_x, 0.5), rect_width, rect_height,
                              facecolor=colors[i], edgecolor='black', linewidth=2,
                              transform=ax6.transAxes)
        ax6.add_patch(rect)
        ax6.text(rect_x + rect_width/2, 0.65, f'头{i+1}',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=ax6.transAxes)

    # 输入
    ax6.text(0.5, 0.9, '输入 X', ha='center', fontsize=12, fontweight='bold',
             transform=ax6.transAxes)
    ax6.annotate('', xy=(0.5, 0.82), xytext=(0.5, 0.88),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2),
                 transform=ax6.transAxes)

    # Concat
    ax6.text(0.5, 0.35, '拼接 + 线性变换', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow'),
             transform=ax6.transAxes)

    # 输出
    ax6.annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.32),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2),
                 transform=ax6.transAxes)
    ax6.text(0.5, 0.15, '输出', ha='center', fontsize=12, fontweight='bold',
             transform=ax6.transAxes)

    ax6.set_title('多头注意力机制', fontsize=12)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap12_fig1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap12_fig1.png")
    print(f"'爱'的注意力权重: {weights_love}")


def create_positional_encoding():
    """
    位置编码可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 位置编码参数
    max_len = 100
    d_model = 64
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    # 计算位置编码 (Sinusoidal)
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    div_term = 10000 ** (2 * (dims // 2) / d_model)
    PE = np.zeros((max_len, d_model))
    PE[:, 0::2] = np.sin(positions / div_term[:, 0::2])
    PE[:, 1::2] = np.cos(positions / div_term[:, 1::2])

    # === 子图1: 位置编码热图 ===
    ax1 = axes[0, 0]
    im1 = ax1.imshow(PE[:50, :32], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('维度', fontsize=11, labelpad=10)
    ax1.set_ylabel('位置', fontsize=11, labelpad=10)
    ax1.set_title('正弦位置编码', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    # === 子图2: 不同维度的sin/cos波形 ===
    ax2 = axes[0, 1]
    positions_plot = np.arange(50)

    for dim in [0, 2, 4, 8, 16]:
        ax2.plot(positions_plot, PE[:50, dim], label=f'维度 {dim}', linewidth=1.5)

    ax2.set_xlabel('位置', fontsize=11, labelpad=10)
    ax2.set_ylabel('编码值', fontsize=11, labelpad=10)
    ax2.set_title('不同维度的位置编码波形', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # === 子图3: 相邻位置的相似度 ===
    ax3 = axes[1, 0]

    # 计算位置0与其他位置的点积
    similarity = PE @ PE[0, :]
    ax3.plot(positions[:50], similarity[:50], 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--')

    ax3.set_xlabel('位置', fontsize=11, labelpad=10)
    ax3.set_ylabel('与位置0的点积', fontsize=11, labelpad=10)
    ax3.set_title('位置相似度: PE[pos] · PE[0]', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # === 子图4: RoPE示意 ===
    ax4 = axes[1, 1]

    # RoPE: 旋转位置编码
    # 对于2维情况，就是一个旋转矩阵
    theta_base = 10000
    d = 8  # 维度

    # 画几个位置的旋转
    for pos in [0, 1, 2, 4, 8]:
        angle = pos / (theta_base ** (np.arange(0, d, 2) / d))
        x = np.cos(angle)
        y = np.sin(angle)
        ax4.scatter(x, y, s=50, label=f'位置={pos}')
        for i in range(len(x)):
            ax4.annotate(f'd={2*i}', (x[i], y[i]), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)

    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 画单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

    ax4.set_xlabel('cos(θ)', fontsize=11, labelpad=10)
    ax4.set_ylabel('sin(θ)', fontsize=11, labelpad=10)
    ax4.set_title('RoPE: 旋转位置编码', fontsize=12)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap12_fig2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap12_fig2.png")


def create_attention_patterns():
    """
    不同类型的注意力模式
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    seq_len = 10

    # === 模式1: 均匀注意力 ===
    ax1 = axes[0]
    uniform = np.ones((seq_len, seq_len)) / seq_len
    im1 = ax1.imshow(uniform, cmap='Blues', vmin=0, vmax=0.5)
    ax1.set_title('均匀注意力', fontsize=12)
    ax1.set_xlabel('键位置', labelpad=10)
    ax1.set_ylabel('查询位置', labelpad=10)

    # === 模式2: 局部注意力 ===
    ax2 = axes[1]
    local = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(max(0, i-2), min(seq_len, i+3)):
            local[i, j] = 1
    local = local / local.sum(axis=1, keepdims=True)
    im2 = ax2.imshow(local, cmap='Blues', vmin=0, vmax=0.5)
    ax2.set_title('局部注意力 (窗口=5)', fontsize=12)
    ax2.set_xlabel('键位置', labelpad=10)

    # === 模式3: Causal (自回归) ===
    ax3 = axes[2]
    causal = np.tril(np.ones((seq_len, seq_len)))
    causal = causal / causal.sum(axis=1, keepdims=True)
    im3 = ax3.imshow(causal, cmap='Blues', vmin=0, vmax=0.5)
    ax3.set_title('因果注意力 (自回归)', fontsize=12)
    ax3.set_xlabel('键位置', labelpad=10)

    # === 模式4: 稀疏注意力 ===
    ax4 = axes[3]
    sparse = np.zeros((seq_len, seq_len))
    # 局部 + 全局
    for i in range(seq_len):
        # 局部
        for j in range(max(0, i-1), min(seq_len, i+2)):
            sparse[i, j] = 1
        # 全局（每隔3个位置）
        for j in range(0, seq_len, 3):
            sparse[i, j] = 1
    sparse = sparse / sparse.sum(axis=1, keepdims=True)
    im4 = ax4.imshow(sparse, cmap='Blues', vmin=0, vmax=0.5)
    ax4.set_title('稀疏注意力', fontsize=12)
    ax4.set_xlabel('键位置', labelpad=10)

    for ax in axes:
        ax.set_xticks(range(0, seq_len, 2))
        ax.set_yticks(range(0, seq_len, 2))

    plt.tight_layout()
    plt.savefig('/home/user/full_book/figs/chap12_fig3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图像已保存到 figs/chap12_fig3.png")


if __name__ == "__main__":
    print("=== 第12章：Transformer可视化 ===\n")

    print("1. 生成自注意力计算可视化...")
    create_attention_visualization()

    print("\n2. 生成位置编码可视化...")
    create_positional_encoding()

    print("\n3. 生成注意力模式对比...")
    create_attention_patterns()

    print("\n所有图像生成完成!")
