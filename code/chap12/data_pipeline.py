#!/usr/bin/env python3
"""
第12章 实验二：最小数据流水线 —— 生成、验证、过滤、训练学生、再用学生生成
==================================================================================

从仓库根目录运行::

    python3 code/chap12/data_pipeline.py        # CPU 单线程约 1 min

输出 figures/chap12/data_pipeline.{pdf,png}，四个面板：
    (a) 硬过滤流水线：每轮的可行率（折线）与筛出的可行样本数（柱）——"数据飞轮"
    (b) 硬过滤 vs 软加权 w = exp(−λ·违约量)（λ=1, 4）：可行率随轮数的上升；虚线是
        "学生完美拟合"假设下的理论预测 p_n ∝ p_0 · w^n
    (c) 硬过滤第 5 轮生成的样本与环形可行域
    (d) 软加权（λ=1）第 5 轮生成的样本与环形可行域

设定：
    生成器  一个小自回归网络，把二维点当成长度为 2 的"句子"：先出 x 的格子号，再
            条件于 x（傅里叶特征）出 y 的格子号；40×40 格覆盖 [−2,2]²，格内均匀抖动。
            初始化时最后一层置零，所以第 0 轮是均匀分布——一个什么都不懂的生成器。
    验证器  程序判定 1.0 ≤ |x| ≤ 1.3（环形可行域），并给出到可行域的距离作为"违约量"。
    过滤器  硬：只留可行样本（λ→∞）；软：样本权重 ∝ exp(−λ·违约量)。
    学生    与生成器同结构的新网络，在（加权的）样本上做极大似然，然后接替生成器。

随机种子固定（0），重跑结果一致。
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, 'code')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

torch.set_num_threads(1)
setup_style()

OUT = 'figures/chap12'
SEED = 0
V = 40                       # 每个坐标的格子数
LO, HI = -2.0, 2.0
CELL = (HI - LO) / V
R_IN, R_OUT = 1.0, 1.3       # 环形可行域
N_SAMPLES = 4000             # 每轮生成的样本数
ROUNDS = 5                   # 迭代轮数
TRAIN_STEPS = 400            # 每轮学生的训练步数


# =============================================================================
# 验证器：程序判定可行性，并给出违约量（到可行域的距离）
# =============================================================================
def violation(xy):
    """到环形可行域 R_IN ≤ |x| ≤ R_OUT 的距离；可行点为 0。"""
    rad = torch.linalg.norm(xy, dim=1)
    return torch.clamp(R_IN - rad, min=0.0) + torch.clamp(rad - R_OUT, min=0.0)


def verify(xy):
    return violation(xy) == 0.0


# =============================================================================
# 生成器 / 学生：长度为 2 的自回归模型
# =============================================================================
def fourier_features(t, n_freq=8):
    """把格子号映射到坐标值，再取正弦/余弦特征（学生因此能在相邻 x 之间泛化）。"""
    x = LO + (t.float() + 0.5) * CELL                     # 格子中心坐标
    freqs = torch.arange(1, n_freq + 1, dtype=torch.float32) * (np.pi / (HI - LO))
    ang = x[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)   # (N, 2 n_freq)


class ARGenerator(nn.Module):
    def __init__(self, n_freq=8, hidden=64):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(V))                  # p(x)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, V))             # p(y | x)
        nn.init.zeros_(self.mlp[-1].weight)                        # 初始 = 均匀分布
        nn.init.zeros_(self.mlp[-1].bias)

    def log_prob(self, idx):
        lp1 = F.log_softmax(self.first, 0)[idx[:, 0]]
        logits2 = self.mlp(fourier_features(idx[:, 0]))
        lp2 = F.log_softmax(logits2, -1).gather(1, idx[:, 1:2]).squeeze(1)
        return lp1 + lp2

    @torch.no_grad()
    def sample(self, n, g):
        i1 = torch.multinomial(F.softmax(self.first, 0), n, replacement=True, generator=g)
        p2 = F.softmax(self.mlp(fourier_features(i1)), -1)
        i2 = torch.multinomial(p2, 1, generator=g).squeeze(1)
        idx = torch.stack([i1, i2], 1)
        xy = LO + (idx.float() + torch.rand(idx.shape, generator=g)) * CELL   # 格内均匀抖动
        return idx, xy

    @torch.no_grad()
    def cell_probs(self):
        """整张 40×40 格的概率表（用于理论预测的对照）。"""
        p1 = F.softmax(self.first, 0)
        p2 = F.softmax(self.mlp(fourier_features(torch.arange(V))), -1)
        return p1[:, None] * p2


def train_student(idx, w, steps=TRAIN_STEPS, lr=1e-2, seed=SEED):
    """在加权样本上做极大似然：min −Σ w_i log p(x_i) / Σ w_i。"""
    torch.manual_seed(seed)
    student = ARGenerator()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    w = w / w.sum()
    for _ in range(steps):
        loss = -(w * student.log_prob(idx)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return student


# =============================================================================
# 流水线
# =============================================================================
def run_pipeline(mode, lam=None, rounds=ROUNDS, seed=SEED):
    """mode='hard'：只留可行样本；mode='soft'：权重 exp(−λ·违约量)。返回每轮统计与末轮样本。"""
    g = torch.Generator().manual_seed(seed)
    gen = ARGenerator()
    rates, n_kept, coverage, ang_ent = [], [], [], []
    feas_cells = feasible_cell_mask()
    xy_last = None
    for rd in range(rounds + 1):
        idx, xy = gen.sample(N_SAMPLES, g)
        ok = verify(xy)
        rates.append(float(ok.float().mean()))
        hit = torch.zeros(V, V, dtype=torch.bool)
        hit[idx[ok, 0], idx[ok, 1]] = True
        coverage.append(float((hit & feas_cells).sum() / feas_cells.sum()))
        ang_ent.append(angular_entropy(xy))
        xy_last = xy
        if rd == rounds:
            break
        if mode == 'hard':
            w = ok.float()
        else:
            w = torch.exp(-lam * violation(xy))
        n_kept.append(int(ok.sum()))
        gen = train_student(idx, w, seed=seed + rd)
    return dict(rates=np.array(rates), n_kept=np.array(n_kept),
                coverage=np.array(coverage), ang_ent=np.array(ang_ent), xy=xy_last.numpy())


# =============================================================================
# 理论预测：学生完美拟合时 p_n ∝ p_0 · w^n（软），或 p_n ∝ p_0 · f^n（硬，f = 格内可行比例）
# =============================================================================
def cell_stats(n_mc=400, seed=SEED):
    """每个格子里可行比例 f 与平均软权重 E[exp(−λ d)]（蒙特卡洛）。"""
    g = torch.Generator().manual_seed(seed + 99)
    ii, jj = torch.meshgrid(torch.arange(V), torch.arange(V), indexing='ij')
    idx = torch.stack([ii.reshape(-1), jj.reshape(-1)], 1)
    idx = idx.repeat_interleave(n_mc, 0)
    xy = LO + (idx.float() + torch.rand(idx.shape, generator=g)) * CELL
    d = violation(xy).view(V * V, n_mc)
    f = (d == 0).float().mean(1).view(V, V)
    return f, d.view(V * V, n_mc)


def angular_entropy(xy, n_bins=36):
    """可行样本在环上的角向分布熵（归一化到 [0,1]，1 = 均匀铺满整个环）——检查有没有模式坍缩。"""
    ok = verify(xy)
    ang = torch.atan2(xy[ok, 1], xy[ok, 0])
    hist = torch.histc(ang, bins=n_bins, min=-np.pi, max=np.pi)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum() / np.log(n_bins))


def feasible_cell_mask():
    f, _ = cell_stats()
    return f > 0.999


def theory_rates(mode, lam, rounds=ROUNDS):
    f, d = cell_stats()
    p = torch.full((V, V), 1.0 / (V * V))
    if mode == 'hard':
        w_cell = f
    else:
        w_cell = torch.exp(-lam * d).mean(1).view(V, V)
    out = []
    for _ in range(rounds + 1):
        out.append(float((p * f).sum()))
        p = p * w_cell
        p = p / p.sum()
    return np.array(out)


# =============================================================================
# 主流程
# =============================================================================
def draw_region(ax):
    ax.add_patch(Circle((0, 0), R_OUT, facecolor=COLORS['green'], alpha=0.18, edgecolor='none'))
    ax.add_patch(Circle((0, 0), R_IN, facecolor='white', alpha=1.0, edgecolor='none'))
    ax.add_patch(Circle((0, 0), R_OUT, fill=False, edgecolor=COLORS['green'], lw=1.2))
    ax.add_patch(Circle((0, 0), R_IN, fill=False, edgecolor=COLORS['green'], lw=1.2))
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect('equal')
    ax.grid(False)


def main():
    t0 = time.time()
    res_hard = run_pipeline('hard')
    res_soft4 = run_pipeline('soft', lam=4.0)
    res_soft1 = run_pipeline('soft', lam=1.0)
    th_hard = theory_rates('hard', None)
    th_soft4 = theory_rates('soft', 4.0)
    th_soft1 = theory_rates('soft', 1.0)

    np.set_printoptions(precision=3, suppress=True)
    print('理论第 0 轮可行率（均匀分布下的环面积比）=', th_hard[0])
    for name, res, th in [('硬过滤', res_hard, th_hard), ('软 λ=4', res_soft4, th_soft4),
                          ('软 λ=1', res_soft1, th_soft1)]:
        print(f'{name}: 可行率 = {res["rates"]}')
        print(f'{"":6s}  理论   = {th}')
        print(f'{"":6s}  覆盖率 = {res["coverage"]}')
        print(f'{"":6s}  角向熵 = {res["ang_ent"]}')
        if name == '硬过滤':
            print(f'{"":6s}  每轮筛出的可行样本数 = {res["n_kept"]}')
    print(f'用时 {time.time()-t0:.1f} s')

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(fig_size(2, 2)[0], fig_size(2, 2)[1] * 1.15))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    rounds = np.arange(ROUNDS + 1)

    # (a) 硬过滤：可行率 + 每轮筛出的样本数
    ax_a2 = ax_a.twinx()
    ax_a2.bar(rounds[:-1] + 0.5, res_hard['n_kept'], width=0.5, color=COLORS['gray'], alpha=0.35,
              label='筛出的可行样本数（训练下一轮学生）')
    ax_a2.set_ylabel('可行样本数', color=COLORS['gray'])
    ax_a2.tick_params(axis='y', colors=COLORS['gray'])
    ax_a2.spines['right'].set_visible(True)
    ax_a2.grid(False)
    ax_a2.set_ylim(0, N_SAMPLES * 1.05)
    ax_a.plot(rounds, res_hard['rates'], 'o-', color=COLORS['blue'], label='生成样本的可行率', zorder=3)
    for rd, rate in zip(rounds, res_hard['rates']):
        ax_a.annotate(f'{rate:.0%}', (rd, rate), textcoords='offset points', xytext=(0, 6),
                      ha='center', fontsize=7.5, color=COLORS['blue'])
    ax_a.set_ylim(0, 1.18)
    ax_a.set_xlabel('轮数')
    ax_a.set_ylabel('可行率（硬过滤）')
    ax_a.set_xticks(rounds)
    ha, la = ax_a.get_legend_handles_labels()
    hb, lb = ax_a2.get_legend_handles_labels()
    ax_a.legend(ha + hb, la + lb, loc='center', fontsize=7.5, frameon=True, framealpha=0.9)
    panel_label(ax_a, '(a)')

    # (b) 硬 vs 软
    for res, th, col, name in [(res_hard, th_hard, COLORS['blue'], r'硬过滤（$\lambda\to\infty$）'),
                               (res_soft4, th_soft4, COLORS['orange'], r'软加权 $\lambda=4$'),
                               (res_soft1, th_soft1, COLORS['red'], r'软加权 $\lambda=1$')]:
        ax_b.plot(rounds, res['rates'], 'o-', color=col, label=name)
        ax_b.plot(rounds, th, ls=':', color=col, lw=1.2)
    ax_b.plot([], [], ls=':', color=COLORS['gray'], label=r'理论 $p_n\propto p_0\,w^n$')
    ax_b.set_ylim(0, 1.05)
    ax_b.set_xlabel('轮数')
    ax_b.set_ylabel('生成样本的可行率')
    ax_b.set_xticks(rounds)
    ax_b.legend(loc='lower right', fontsize=7.5)
    panel_label(ax_b, '(b)')

    # (c) 硬过滤末轮样本
    draw_region(ax_c)
    xy = res_hard['xy'][:1500]
    ok = verify(torch.from_numpy(xy)).numpy()
    ax_c.scatter(xy[ok, 0], xy[ok, 1], s=3, color=COLORS['blue'], alpha=0.6, lw=0, label='可行')
    ax_c.scatter(xy[~ok, 0], xy[~ok, 1], s=4, color=COLORS['red'], alpha=0.8, lw=0, label='不可行')
    ax_c.set_xlabel(r'$x$')
    ax_c.set_ylabel(r'$y$')
    ax_c.legend(loc='upper left', fontsize=7.5, markerscale=2.5, frameon=True, framealpha=0.9)
    ax_c.text(0.97, 0.03, f'硬过滤，第 {ROUNDS} 轮\n可行率 {res_hard["rates"][-1]:.0%}', transform=ax_c.transAxes,
              ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))
    panel_label(ax_c, '(c)')

    # (d) 软加权 λ=1 末轮样本
    draw_region(ax_d)
    xy = res_soft1['xy'][:1500]
    ok = verify(torch.from_numpy(xy)).numpy()
    ax_d.scatter(xy[ok, 0], xy[ok, 1], s=3, color=COLORS['blue'], alpha=0.6, lw=0, label='可行')
    ax_d.scatter(xy[~ok, 0], xy[~ok, 1], s=4, color=COLORS['red'], alpha=0.8, lw=0, label='不可行')
    ax_d.set_xlabel(r'$x$')
    ax_d.set_ylabel(r'$y$')
    ax_d.legend(loc='upper left', fontsize=7.5, markerscale=2.5, frameon=True, framealpha=0.9)
    ax_d.text(0.97, 0.03, f'软加权 $\\lambda=1$，第 {ROUNDS} 轮\n可行率 {res_soft1["rates"][-1]:.0%}', transform=ax_d.transAxes,
              ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))
    panel_label(ax_d, '(d)')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUT, 'data_pipeline'))


if __name__ == '__main__':
    main()
