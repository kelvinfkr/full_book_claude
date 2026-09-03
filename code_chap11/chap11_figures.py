#!/usr/bin/env python3
"""
第11章：注意力机制与位置编码
==============================

从仓库根目录运行::

    python3 code_chap11/chap11_figures.py

输出（每张同时有 pdf 与 png）：

    figs_chap11/chap11_fig1   注意力机制：在含噪正弦序列上训练的单头自注意力
    figs_chap11/chap11_fig2   正弦位置编码：编码矩阵、各维度波形、位置间点积

图 1 的思路
-----------
用一个物理时间序列做"真实的小例子"：含噪的正弦信号 x_t = sin(2π t/16 + φ) + ε_t，
序列长 48（3 个周期）。模型只有一层单头自注意力：
    q_t = W_q f_t,  k_t = W_k f_t,  v_t = w_v·f_t + b,
    α_tj = softmax_j(q_t·k_j / √d_k),  y_t = Σ_j α_tj v_j，
其中 f_t = (x_t, x_{t-1}, x_{t-2}, x_{t-3}) 是最近 4 个采样值（延迟嵌入，
它决定了该时刻的"相位"）。训练目标：输出 y_t 逼近无噪声的 sin。
最优的去噪方式是把所有"相位相同"的时刻的观测平均起来，于是模型学出的
注意力矩阵呈现周期为 16 的条纹——这正是第11章说的"注意力 = softmax 核回归"。
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, 'code')
from textbook_style import setup_style, save_figure, panel_label, COLORS  # noqa: E402

setup_style()
torch.manual_seed(0)
torch.set_num_threads(1)  # 这么小的模型，多线程反而因同步开销慢上百倍

# =============================================================================
# 图 1：在含噪正弦序列上训练单头自注意力
# =============================================================================
T, PERIOD, SIGMA, N_LAG, D_K = 48, 16, 0.3, 4, 8


def make_batch(rng, batch):
    """随机相位的正弦 + 高斯噪声；返回 (含噪输入, 干净目标)，形状 (B, T)。"""
    phase = rng.uniform(0, 2 * np.pi, size=(batch, 1))
    t = np.arange(T)[None, :]
    clean = np.sin(2 * np.pi * t / PERIOD + phase)
    noisy = clean + SIGMA * rng.standard_normal(clean.shape)
    return noisy.astype(np.float32), clean.astype(np.float32)


def lag_features(x, n_lag=N_LAG):
    """f_t = (x_t, x_{t-1}, ..., x_{t-n_lag+1})，序列开头用 x_0 填充。"""
    feats = [torch.roll(x, shifts=lag, dims=1) for lag in range(n_lag)]
    for lag in range(1, n_lag):
        feats[lag][:, :lag] = x[:, :1]
    return torch.stack(feats, dim=-1)


class SingleHeadAttention(nn.Module):
    def __init__(self, d_in, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_k, bias=False)
        self.Wk = nn.Linear(d_in, d_k, bias=False)
        self.Wv = nn.Linear(d_in, 1)
        self.d_k = d_k

    def forward(self, f):
        q, k, v = self.Wq(f), self.Wk(f), self.Wv(f)
        scores = q @ k.transpose(1, 2) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        return (attn @ v).squeeze(-1), attn


def train_attention(steps=2000, batch=128, lr=5e-3, seed=0):
    rng = np.random.default_rng(seed)
    model = SingleHeadAttention(N_LAG, D_K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        noisy, clean = make_batch(rng, batch)
        x, y_true = torch.from_numpy(noisy), torch.from_numpy(clean)
        y_pred, _ = model(lag_features(x))
        loss = ((y_pred - y_true) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1:4d}  训练 MSE = {loss.item():.4f}")
    return model


print("=" * 60)
print("图 1：训练单头自注意力去噪含噪正弦序列")
print("=" * 60)
model = train_attention()

# 测试序列：固定相位，便于讨论"哪些时刻同相位"
rng_test = np.random.default_rng(123)
noisy_test, clean_test = make_batch(rng_test, 1)
with torch.no_grad():
    y_out, attn = model(lag_features(torch.from_numpy(noisy_test)))
y_out, A = y_out[0].numpy(), attn[0].numpy()
mse_in = np.mean((noisy_test[0] - clean_test[0]) ** 2)
mse_out = np.mean((y_out - clean_test[0]) ** 2)
print(f"测试序列：输入噪声 MSE = {mse_in:.4f}，注意力输出 MSE = {mse_out:.4f}")

# 大规模统计：注意力输出与"同相位平均"的理想去噪相比
noisy_big, clean_big = make_batch(np.random.default_rng(7), 512)
with torch.no_grad():
    y_big, A_big = model(lag_features(torch.from_numpy(noisy_big)))
print(f"512 条测试序列：输入 MSE = {np.mean((noisy_big - clean_big) ** 2):.4f}，"
      f"输出 MSE = {np.mean((y_big.numpy() - clean_big) ** 2):.4f}")
Q_POS = 24
same_phase = [j for j in range(T) if (j - Q_POS) % PERIOD == 0]
row = A[Q_POS]
print(f"查询位置 t={Q_POS} 的注意力：同相位位置 {same_phase} 的权重之和 = {row[same_phase].sum():.2f}，"
      f"相位相差 ±3 以内的权重之和 = "
      f"{sum(row[j] for j in range(T) if min((j - Q_POS) % PERIOD, (Q_POS - j) % PERIOD) <= 3):.2f}，"
      f"最大权重 = {row.max():.2f}（均匀分布应为 {1 / T:.3f}）")

fig = plt.figure(figsize=(6.3, 5.2))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0], width_ratios=[1.0, 1.0],
                      left=0.09, right=0.98, top=0.95, bottom=0.09, hspace=0.5, wspace=0.42)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# (a) 输入序列、真实信号、注意力输出
ax = ax_a
t = np.arange(T)
ax.plot(t, noisy_test[0], 'o-', color=COLORS['gray'], lw=0.8, markersize=2.8, alpha=0.8,
        label=r'含噪输入 $x_t$')
ax.plot(t, clean_test[0], '-', color=COLORS['black'], lw=1.4, label=r'无噪声信号 $\sin(2\pi t/16+\phi)$')
ax.plot(t, y_out, '-', color=COLORS['blue'], lw=1.6, label=r'注意力输出 $y_t=\sum_j \alpha_{tj} v_j$')
ax.axvline(Q_POS, color=COLORS['red'], ls='--', lw=1.0)
ax.text(Q_POS + 0.5, 1.45, f'查询位置 $t={Q_POS}$', color=COLORS['red'], fontsize=8, va='top')
for j in same_phase:
    if j != Q_POS:
        ax.axvline(j, color=COLORS['red'], ls=':', lw=0.9, alpha=0.7)
        ax.text(j + 0.5, 1.45, f'同相位 $t={j}$', color=COLORS['red'], fontsize=7.5, va='top', alpha=0.85)
ax.set_xlim(-0.5, T - 0.5)
ax.set_ylim(-2.35, 1.7)
ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r'时刻 $t$', labelpad=1)
ax.set_ylabel(r'$x_t$')
ax.legend(fontsize=7.5, loc='lower left', ncol=3, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
panel_label(ax, '(a)', x=-0.04)

# (b) 注意力矩阵热图（每行和为 1）
ax = ax_b
im = ax.imshow(A, cmap='Blues', interpolation='nearest', vmin=0, vmax=A.max())
ax.axhline(Q_POS, color=COLORS['red'], lw=0.9, ls='--')
ax.set_xlabel(r'键位置 $j$', labelpad=1)
ax.set_ylabel(r'查询位置 $i$')
ax.set_xticks([0, 8, 16, 24, 32, 40])
ax.set_yticks([0, 8, 16, 24, 32, 40])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.set_title(r'$\alpha_{ij}$', fontsize=9, pad=4)
cb.ax.tick_params(labelsize=7.5)
panel_label(ax, '(b)', x=-0.2)

# (c) 查询位置 t=24 的注意力分布
ax = ax_c
colors = [COLORS['red'] if j in same_phase else COLORS['blue'] for j in range(T)]
ax.bar(t, row, color=colors, width=0.8)
ax.axvline(Q_POS, color=COLORS['red'], ls='--', lw=0.9)
ax.axhline(1 / T, color=COLORS['gray'], ls=':', lw=1.0)
ax.text(T - 0.5, row.max() * 1.22, f'灰色点线：均匀分布 $1/{T}$', fontsize=7, color=COLORS['gray'],
        ha='right', va='top')
for j in same_phase:
    ax.annotate(f'$j={j}$', xy=(j, row[j]), xytext=(0, 3), textcoords='offset points',
                fontsize=7.5, color=COLORS['red'], ha='center')
ax.set_xlim(-0.5, T - 0.5)
ax.set_ylim(0, row.max() * 1.25)
ax.set_xticks([0, 8, 16, 24, 32, 40])
ax.set_xlabel(r'键位置 $j$', labelpad=1)
ax.set_ylabel(rf'$\alpha_{{{Q_POS},\,j}}$')
panel_label(ax, '(c)', x=-0.2)

save_figure(fig, 'figs_chap11/chap11_fig1')

# =============================================================================
# 图 2：正弦位置编码
# =============================================================================
print("\n" + "=" * 60)
print("图 2：正弦位置编码")
print("=" * 60)
N_POS, D_MODEL, BASE = 64, 16, 10000.0


def sinusoidal_pe(n_pos, d_model, base=BASE):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_model // 2)[None, :]
    ang = pos / base ** (2 * i / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


PE = sinusoidal_pe(N_POS, D_MODEL)
wavelengths = 2 * np.pi * BASE ** (2 * np.arange(D_MODEL // 2) / D_MODEL)
print("各维度对（2i, 2i+1）的波长：", np.array2string(wavelengths, precision=1, max_line_width=200))
S = PE @ PE.T / (D_MODEL / 2)  # 归一化点积：对角线为 1
print(f"点积矩阵：对角线 = {S[0, 0]:.2f}，相邻位置 = {S[0, 1]:.2f}，相隔 8 = {S[0, 8]:.2f}，"
      f"相隔 32 = {S[0, 32]:.2f}；Toeplitz 检验 max|S[i,j]-S[i+1,j+1]| = "
      f"{np.abs(S[:-1, :-1] - S[1:, 1:]).max():.1e}")

fig = plt.figure(figsize=(6.3, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[0.72, 1.3], height_ratios=[1.0, 1.15],
                      left=0.08, right=0.97, top=0.93, bottom=0.10, hspace=0.6, wspace=0.45)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# (a) 编码矩阵：位置 × 维度
ax = ax_a
im = ax.imshow(PE, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1, aspect='auto')
ax.set_xlabel(r'维度 $d$', labelpad=1)
ax.set_ylabel(r'位置 $pos$')
ax.set_xticks([0, 4, 8, 12, 15])
ax.set_yticks([0, 16, 32, 48, 63])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.09, pad=0.05)
cb.set_label(r'$PE_{(pos,\,d)}$', fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
panel_label(ax, '(a)', x=-0.16)

# (b) 几个维度的正弦曲线随位置变化
ax = ax_b
pos_fine = np.linspace(0, N_POS - 1, 1200)
show_dims = [0, 2, 4, 6]
cols = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]
for d, col in zip(show_dims, cols):
    lam = wavelengths[d // 2]
    ax.plot(pos_fine, np.sin(2 * np.pi * pos_fine / lam), color=col, lw=1.3,
            label=rf'$d={d}$（波长 {lam:.0f}）' if lam >= 10 else rf'$d={d}$（波长 {lam:.1f}）')
    ax.plot(np.arange(N_POS), PE[:, d], 'o', color=col, markersize=1.8)
ax.set_xlim(0, N_POS - 1)
ax.set_ylim(-1.25, 1.25)
ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r'位置 $pos$', labelpad=1)
ax.set_ylabel(r'$PE_{(pos,\,d)}$')
ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.2,
          columnspacing=0.8, borderaxespad=0.0)
panel_label(ax, '(b)', x=-0.12, y=1.18)

# (c) 位置之间的点积：只依赖相对位置
ax = ax_c
im = ax.imshow(S, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
ax.set_xlabel(r"位置 $pos'$", labelpad=1)
ax.set_ylabel(r'位置 $pos$')
ax.set_xticks([0, 16, 32, 48, 63])
ax.set_yticks([0, 16, 32, 48, 63])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"$PE_{pos}\cdot PE_{pos'}\,/\,(d/2)$", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
panel_label(ax, '(c)', x=-0.12)

save_figure(fig, 'figs_chap11/chap11_fig2')
print("\n第11章图生成完成！")
