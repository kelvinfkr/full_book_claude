#!/usr/bin/env python3
"""
第12章 实验一：玩具 RLHF —— 三种后训练方法都在逼近同一个玻尔兹曼分布
==================================================================

从仓库根目录运行::

    python3 code_chap12/toy_rlhf.py          # CPU 单线程约 20 s

输出 figs_chap12/toy_rlhf.{pdf,png}，四个面板：
    (a) 问题的输入：K=10 个"回答"的奖励 r_k 与参考策略 π_ref
    (b) 闭式解 π*(y) ∝ π_ref(y) exp(r(y)/β) 在 β∈{0.1, 0.5, 2} 下的形状（温度效应）
    (c) 三种方法训练出的策略与 π*（β=0.5）的 KL 距离随迭代下降
    (d) 训练结束时三种策略与 π* 的逐动作对比

"语言模型"被压缩成一个只有 K=10 个动作的 softmax 策略 π_θ(y) = softmax(θ)_y：
一个 prompt、K 种可能的回答。三种方法：
    PPO-KL ：InstructGPT（2022）式——KL 惩罚写进奖励，裁剪的重要性比，采样估计梯度
    DPO    ：Rafailov 等（2023）——用 Bradley–Terry 偏好对直接学策略，不学奖励模型
    GRPO   ：Shao 等（2024）——同一 prompt 采样一组回答，组内均值作基线，KL 项用 k3 估计
三者的目标都是 max_π E_π[r] − β KL(π‖π_ref)，闭式解 π* 就是第4章的玻尔兹曼分布。

随机种子固定（0），重跑结果一致。
"""
import os
import sys
import time

# 单线程 BLAS（须在 import numpy/torch 之前设置）
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, 'code')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

torch.set_num_threads(1)
setup_style()

OUT = 'figs_chap12'
K = 10          # 动作（"回答"）个数
BETA = 0.5      # 训练时使用的 KL 系数
SEED = 0
ITERS = 300     # 三种方法都跑 300 次参数更新


# =============================================================================
# 问题设定
# =============================================================================
def make_problem(seed=SEED):
    """随机生成参考策略 π_ref 与奖励向量 r。"""
    g = torch.Generator().manual_seed(seed)
    logits_ref = torch.randn(K, generator=g)
    pi_ref = torch.softmax(logits_ref, 0)
    r = torch.rand(K, generator=g)          # 奖励落在 [0, 1]
    return pi_ref, r


def closed_form(pi_ref, r, beta):
    """π*(y) ∝ π_ref(y) exp(r(y)/β) —— 玻尔兹曼分布，β 是温度。"""
    return torch.softmax(torch.log(pi_ref) + r / beta, 0)


def kl(p, q):
    """KL(p‖q)，p、q 是长度 K 的概率向量。"""
    return float((p * (torch.log(p) - torch.log(q))).sum())


def lr_at(it, lr0, iters):
    """学习率线性衰减到 10%，与 PPO 实践里的线性退火一致，也让采样噪声不至于把策略一直抖来抖去。"""
    return lr0 * (1.0 - 0.9 * it / iters)


# =============================================================================
# 方法 (a)：PPO-KL —— KL 惩罚写进奖励，裁剪的重要性比
# =============================================================================
def train_ppo_kl(pi_ref, r, beta, pi_star, iters=ITERS, n=256, lr=0.05,
                 clip=0.2, epochs=4, seed=SEED):
    g = torch.Generator().manual_seed(seed + 10)
    theta = torch.log(pi_ref).clone().requires_grad_(True)     # 从参考策略出发
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for it in range(iters):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(it, lr, iters)
        with torch.no_grad():
            pi_old = torch.softmax(theta, 0)
            y = torch.multinomial(pi_old, n, replacement=True, generator=g)
            logp_old = torch.log(pi_old[y])
            # KL 惩罚进奖励：r̃(y) = r(y) − β [log π_old(y) − log π_ref(y)]
            r_tilde = r[y] - beta * (logp_old - torch.log(pi_ref[y]))
            adv = r_tilde - r_tilde.mean()                       # 批均值作基线
        for _ in range(epochs):
            logp = torch.log_softmax(theta, 0)[y]
            ratio = torch.exp(logp - logp_old)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            loss = -surr.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append(kl(torch.softmax(theta.detach(), 0), pi_star))
    return torch.softmax(theta.detach(), 0), np.array(hist)


# =============================================================================
# 方法 (b)：DPO —— 偏好对来自 π_ref 的采样，标签服从 Bradley–Terry(r)
# =============================================================================
def make_preferences(pi_ref, r, n_pairs, seed):
    g = torch.Generator().manual_seed(seed + 20)
    y1 = torch.multinomial(pi_ref, n_pairs, replacement=True, generator=g)
    y2 = torch.multinomial(pi_ref, n_pairs, replacement=True, generator=g)
    keep = y1 != y2
    y1, y2 = y1[keep], y2[keep]
    p_first = torch.sigmoid(r[y1] - r[y2])          # P(y1 ≻ y2) = σ(r(y1) − r(y2))
    first_wins = torch.rand(len(y1), generator=g) < p_first
    yw = torch.where(first_wins, y1, y2)
    yl = torch.where(first_wins, y2, y1)
    return yw, yl


def train_dpo(pi_ref, r, beta, pi_star, iters=ITERS, n_pairs=4000, lr=0.05, seed=SEED):
    yw, yl = make_preferences(pi_ref, r, n_pairs, seed)
    theta = torch.log(pi_ref).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    log_ref = torch.log(pi_ref)
    hist = []
    for it in range(iters):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(it, lr, iters)
        logp = torch.log_softmax(theta, 0)
        # 隐式奖励 r̂(y) = β [log π_θ(y) − log π_ref(y)]，配分函数在差里消掉
        h = beta * ((logp[yw] - log_ref[yw]) - (logp[yl] - log_ref[yl]))
        loss = -F.logsigmoid(h).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(kl(torch.softmax(theta.detach(), 0), pi_star))
    return torch.softmax(theta.detach(), 0), np.array(hist), len(yw)


# =============================================================================
# 方法 (c)：GRPO —— 组采样 + 组内均值基线 + k3 估计的 KL 项
# =============================================================================
def train_grpo(pi_ref, r, beta, pi_star, iters=ITERS, groups=16, G=16, lr=0.05,
               clip=0.2, epochs=4, seed=SEED):
    g = torch.Generator().manual_seed(seed + 30)
    theta = torch.log(pi_ref).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    log_ref = torch.log(pi_ref)
    hist = []
    for it in range(iters):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(it, lr, iters)
        with torch.no_grad():
            pi_old = torch.softmax(theta, 0)
            y = torch.multinomial(pi_old, groups * G, replacement=True, generator=g).view(groups, G)
            logp_old = torch.log(pi_old[y])
            rew = r[y]
            adv = rew - rew.mean(dim=1, keepdim=True)            # 组内均值作基线（不需要价值网络）
        for _ in range(epochs):
            logp = torch.log_softmax(theta, 0)[y]
            ratio = torch.exp(logp - logp_old)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            lr_ref = log_ref[y] - logp                           # log π_ref/π_θ
            k3 = torch.exp(lr_ref) - lr_ref - 1.0                # k3 估计的 KL(π_θ‖π_ref)，逐样本非负
            loss = -(surr - beta * k3).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append(kl(torch.softmax(theta.detach(), 0), pi_star))
    return torch.softmax(theta.detach(), 0), np.array(hist)


def grpo_fixed_point(pi_ref, r, beta):
    """k3 估计器逐样本求导，其期望恰是 ∇KL(π_ref‖π_θ)（反向）；
    因此 GRPO 的不动点是 max_π E_π[r] − β KL(π_ref‖π)，这里用精确期望求出它，用作对照线。"""
    theta = torch.log(pi_ref).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for _ in range(3000):
        logp = torch.log_softmax(theta, 0)
        pi = torch.exp(logp)
        obj = (pi * r).sum() - beta * (pi_ref * (torch.log(pi_ref) - logp)).sum()
        opt.zero_grad()
        (-obj).backward()
        opt.step()
    return torch.softmax(theta.detach(), 0)


# =============================================================================
# 主流程
# =============================================================================
def main():
    t0 = time.time()
    pi_ref, r = make_problem()
    pi_star = closed_form(pi_ref, r, BETA)
    betas = [0.1, 0.5, 2.0]
    pi_stars = {b: closed_form(pi_ref, r, b) for b in betas}

    pi_ppo, h_ppo = train_ppo_kl(pi_ref, r, BETA, pi_star)
    pi_dpo, h_dpo, n_pairs = train_dpo(pi_ref, r, BETA, pi_star)
    pi_grpo, h_grpo = train_grpo(pi_ref, r, BETA, pi_star)
    pi_grpo_fp = grpo_fixed_point(pi_ref, r, BETA)

    # ---------------- 数字汇报 ----------------
    np.set_printoptions(precision=3, suppress=True)
    print('奖励 r      =', r.numpy())
    print('π_ref       =', pi_ref.numpy())
    for b in betas:
        p = pi_stars[b]
        print(f'π*(β={b:<3}) =', p.numpy(), f'  熵={float(-(p*torch.log(p)).sum()):.3f}',
              f'  E[r]={float((p*r).sum()):.3f}  KL(π*‖π_ref)={kl(p, pi_ref):.3f}')
    print(f'KL(π_ref‖π*)  起点 = {kl(pi_ref, pi_star):.4f}')
    for name, p, h in [('PPO-KL', pi_ppo, h_ppo), ('DPO', pi_dpo, h_dpo), ('GRPO', pi_grpo, h_grpo)]:
        print(f'{name:7s} 末 KL(π_θ‖π*) = {h[-1]:.2e}   E[r] = {float((p*r).sum()):.3f}   '
              f'KL(π_θ‖π_ref) = {kl(p, pi_ref):.3f}')
    print(f'DPO 偏好对数 = {n_pairs}')
    print(f'GRPO(k3) 理论不动点与 π* 的 KL = {kl(pi_grpo_fp, pi_star):.2e}')
    print(f'用时 {time.time()-t0:.1f} s')

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=fig_size(2, 2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    idx = np.arange(K)
    labels = [str(k) for k in idx]

    # (a) 奖励与参考策略
    ax_a.bar(idx, r.numpy(), color=COLORS['gray'], alpha=0.55, width=0.6, label=r'奖励 $r(y)$')
    ax_a.set_ylabel(r'奖励 $r(y)$')
    ax_a2 = ax_a.twinx()
    ax_a2.plot(idx, pi_ref.numpy(), 'o-', color=COLORS['blue'], label=r'参考策略 $\pi_{\rm ref}(y)$')
    ax_a2.set_ylabel(r'$\pi_{\rm ref}(y)$', color=COLORS['blue'])
    ax_a2.tick_params(axis='y', colors=COLORS['blue'])
    ax_a2.spines['right'].set_visible(True)
    ax_a2.grid(False)
    ax_a2.set_ylim(0, max(pi_ref.numpy()) * 1.45)
    ax_a.set_xticks(idx)
    ax_a.set_xticklabels(labels)
    ax_a.set_xlabel(r'动作 $y$（一个 prompt 的 10 种回答）')
    ax_a.set_ylim(0, 1.45)
    ha, la = ax_a.get_legend_handles_labels()
    hb, lb = ax_a2.get_legend_handles_labels()
    ax_a.legend(ha + hb, la + lb, loc='upper right', ncol=2, fontsize=7.5)
    panel_label(ax_a, '(a)')

    # (b) 温度效应
    w = 0.25
    cols = [COLORS['red'], COLORS['orange'], COLORS['teal']]
    for j, b in enumerate(betas):
        ax_b.bar(idx + (j - 1) * w, pi_stars[b].numpy(), width=w, color=cols[j],
                 label=rf'$\pi^*$, $\beta={b:g}$')
    ax_b.plot(idx, pi_ref.numpy(), 'o', color=COLORS['black'], ms=3.5, label=r'$\pi_{\rm ref}$')
    ax_b.set_xticks(idx)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlabel(r'动作 $y$')
    ax_b.set_ylabel(r'$\pi^*(y)\propto\pi_{\rm ref}(y)\,e^{r(y)/\beta}$')
    ax_b.set_ylim(0, 0.72)
    ax_b.legend(loc='upper center', ncol=2, fontsize=7.5)
    panel_label(ax_b, '(b)')

    # (c) KL 收敛曲线
    its = np.arange(1, ITERS + 1)
    floor = 1e-7                                  # 浮点舍入会让 KL 出现 ~1e-8 甚至负值，对数轴上截断
    ax_c.semilogy(its, np.maximum(h_ppo, floor), color=COLORS['blue'], label='PPO-KL（采样策略梯度）')
    ax_c.semilogy(its, np.maximum(h_dpo, floor), color=COLORS['orange'], label=f'DPO（{n_pairs} 个偏好对）')
    ax_c.semilogy(its, np.maximum(h_grpo, floor), color=COLORS['green'], label='GRPO（组均值基线 + k3）')
    ax_c.axhline(kl(pi_grpo_fp, pi_star), color=COLORS['green'], ls=':', lw=1.2,
                 label='GRPO(k3) 的理论不动点')
    ax_c.set_ylim(floor, 1.0)
    ax_c.set_xlabel('迭代次数')
    ax_c.set_ylabel(r'$\mathrm{KL}(\pi_\theta\,\|\,\pi^*)$，$\beta=0.5$')
    ax_c.legend(loc='lower left', fontsize=7.5)
    panel_label(ax_c, '(c)')

    # (d) 最终策略对比
    w = 0.22
    ax_d.bar(idx - 1.5 * w, pi_star.numpy(), width=w, color=COLORS['black'], label=r'闭式解 $\pi^*$')
    ax_d.bar(idx - 0.5 * w, pi_ppo.numpy(), width=w, color=COLORS['blue'], label='PPO-KL')
    ax_d.bar(idx + 0.5 * w, pi_dpo.numpy(), width=w, color=COLORS['orange'], label='DPO')
    ax_d.bar(idx + 1.5 * w, pi_grpo.numpy(), width=w, color=COLORS['green'], label='GRPO')
    ax_d.plot(idx, pi_ref.numpy(), 'o', color=COLORS['gray'], ms=3.5, label=r'$\pi_{\rm ref}$')
    ax_d.set_xticks(idx)
    ax_d.set_xticklabels(labels)
    ax_d.set_xlabel(r'动作 $y$')
    ax_d.set_ylabel(r'训练结束时的 $\pi_\theta(y)$')
    ax_d.set_ylim(0, 0.52)
    ax_d.legend(loc='upper right', ncol=2, fontsize=7.5)
    panel_label(ax_d, '(d)')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUT, 'toy_rlhf'))


if __name__ == '__main__':
    main()
