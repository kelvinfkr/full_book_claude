"""
罚系数 rho（损失权重）的影响实验（第14章，图 figs_chap14/lambda_effect.pdf）
=====================================================================

问题：一维泊松方程  -u''(x) = pi^2 sin(pi x),  u(0) = u(1) = 0，精确解 u = sin(pi x)。
PINN 的固定权重损失  L = L_PDE + rho * L_BC  就是第1章的罚函数法 f + (rho/2)|g|^2：
rho 是"边界条件这条约束的罚款单价"，不是拉格朗日乘子。

思路：rho 太小 -> 边界条件"不值钱"，训练前期解在端点漂移；
      rho 太大 -> 损失被边界项主导，PDE 残差学得慢、曲线出现锯齿；
      训练足够久之后大家都能收敛，差别主要在"收敛速度"和"稳定性"。
      本脚本对 rho ∈ {0.1, 1, 10, 100, 1000} × 3 个随机种子各训练 10000 轮，画
      (a) 训练早期（第 1000 轮）的逐点误差 |u_theta - u|——看机制：rho 小时误差是一条
          "直线"（边界条件没学到，解整体漂移），rho 大时误差已经很小；
      (b) 整体 MSE 随训练轮数的变化（3 个种子取中位数）——看速度与稳定性：
          rho 大时后期震荡，精度反而上不去；
      (c) 训练结束时的 MSE 与边界误差随 rho 的变化（中位数，阴影为种子间的最小/最大值）。
训练结果缓存在 CACHE 文件里，改图不必重训。

用法：cd 仓库根目录 && python3 code_chap14/lambda_effect.py
（文件名沿用 lambda_effect 以保持与正文 \\includegraphics 的引用一致；正文中该权重记作 rho。）
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from textbook_style import setup_style, save_figure, panel_label, fig_size, COLORS

setup_style()
torch.manual_seed(42); np.random.seed(42)

RHO_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEEDS = [0, 1, 2]
EPOCHS = 10000
SNAP_EPOCH = 1000      # 画"早期快照"的轮数
LR = 1e-3
CACHE = os.environ.get('LAMBDA_EFFECT_CACHE', '/tmp/lambda_effect_cache.npz')


class PINN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)



X_TEST = torch.linspace(0, 1, 200).reshape(-1, 1)
U_EXACT = np.sin(np.pi * X_TEST.numpy()).ravel()


def train(rho, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_int = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])
    mse_hist, snap = [], None
    for ep in range(epochs + 1):
        if ep % 100 == 0:
            with torch.no_grad():
                u_pred = model(X_TEST).numpy().ravel()
            mse_hist.append(float(np.mean((u_pred - U_EXACT) ** 2)))
            if ep == SNAP_EPOCH:
                snap = u_pred.copy()
        if ep == epochs:
            break
        xg = x_int.clone().requires_grad_(True)
        u = model(xg)
        u_x = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xg, torch.ones_like(u_x), create_graph=True)[0]
        f = (np.pi ** 2) * torch.sin(np.pi * xg)
        loss_pde = torch.mean((u_xx + f) ** 2)          # -u'' = f  <=>  u'' + f = 0
        loss_bc = torch.mean(model(x_bc) ** 2)
        loss = loss_pde + rho * loss_bc
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        bc_err = 0.5 * (abs(model(torch.tensor([[0.0]])).item()) + abs(model(torch.tensor([[1.0]])).item()))
    return dict(mse_hist=np.array(mse_hist), snap=snap, mse=mse_hist[-1], bc_err=float(bc_err))


def run_all():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z['runs'].item()
    t0 = time.time()
    runs = {}
    for rho in RHO_VALUES:
        runs[rho] = [train(rho, sd) for sd in SEEDS]
        print(f'rho={rho:7.1f}  MSE(中位数)={np.median([r["mse"] for r in runs[rho]]):.2e}  '
              f'边界误差(中位数)={np.median([r["bc_err"] for r in runs[rho]]):.2e}  ({time.time()-t0:.0f}s)', flush=True)
    np.savez(CACHE, runs=np.array(runs, dtype=object))
    return runs


def main():
    runs = run_all()
    palette = [COLORS['purple'], COLORS['blue'], COLORS['teal'], COLORS['green'], COLORS['orange']]
    fig, axes = plt.subplots(1, 3, figsize=fig_size(3, aspect=0.95))
    x = X_TEST.numpy().ravel()

    # (a) 早期逐点误差：机制
    ax = axes[0]
    for rho, c in zip(RHO_VALUES, palette):
        err = np.abs(runs[rho][0]['snap'] - U_EXACT) + 1e-12
        ax.semilogy(x, err, color=c, lw=1.4, label=f'$\\rho={rho:g}$')
    ax.set_xlabel('$x$'); ax.set_ylabel(f'第 {SNAP_EPOCH} 轮的逐点误差 $|u_\\theta-u|$')
    panel_label(ax, '(a)')

    # (b) MSE 随轮数：速度与稳定性
    ax = axes[1]
    ep = np.arange(0, EPOCHS + 1, 100)
    for rho, c in zip(RHO_VALUES, palette):
        med = np.median(np.stack([r['mse_hist'] for r in runs[rho]]), axis=0)
        ax.semilogy(ep, med, color=c, lw=1.2)
    ax.axvline(SNAP_EPOCH, color=COLORS['gray'], ls=':', lw=1)
    ax.text(SNAP_EPOCH * 1.15, 0.5, '(a) 的时刻', fontsize=7.5, color=COLORS['gray'])
    ax.set_xlabel('训练轮数'); ax.set_ylabel('MSE（3 个种子的中位数）')
    panel_label(ax, '(b)')

    # (c) 最终误差随 rho：精度
    ax = axes[2]
    for key, col, mk, lab in [('bc_err', COLORS['red'], 's', '边界误差'), ('mse', COLORS['blue'], 'o', '整体 MSE')]:
        vals = np.array([[r[key] for r in runs[rho]] for rho in RHO_VALUES])
        med = np.median(vals, axis=1)
        ax.plot(RHO_VALUES, med, marker=mk, color=col, lw=1.4, label=lab)
        ax.fill_between(RHO_VALUES, vals.min(axis=1), vals.max(axis=1), color=col, alpha=0.15, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('罚系数 $\\rho$'); ax.set_ylabel(f'第 {EPOCHS} 轮的误差')
    ax.legend(fontsize=8, loc='lower right')
    panel_label(ax, '(c)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=8.5,
               handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, 'figs_chap14/lambda_effect')

    out = {str(rho): dict(mse=[r['mse'] for r in runs[rho]], bc_err=[r['bc_err'] for r in runs[rho]],
                          mse_at_snap=[float(r['mse_hist'][SNAP_EPOCH // 100]) for r in runs[rho]])
           for rho in RHO_VALUES}
    with open('figs_chap14/lambda_effect_results.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
