"""
PPO with Global Cooperation Reward — Spatial Prisoner's Dilemma
===============================================================
核心设计：
  奖励 = α × 自身收益(归一化) + (1-α) × 全局合作率

  α=1.0 → 纯自私，合作崩溃
  α=0.0 → 纯集体，合作涌现
  α*    → 临界点，论文核心发现

研究问题：
  1. 是否存在临界α*，低于它合作涌现，高于它合作崩溃？
  2. 不同b值下α*如何变化？

网格：50×50，von Neumann邻域，同步更新
优化：CUDA + 大batch + 向量化环境
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

# ── 设备（优先CUDA）──
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ MPS")
else:
    DEVICE = torch.device("cpu")
    print("⚠ CPU")

# ── 超参数 ──
GRID_N       = 50
N_AGENTS     = GRID_N * GRID_N          # 2500
T_STEPS      = 3000
N_SEEDS      = 1
INIT_COOP    = 0.5

LR           = 1e-4
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF   = 0.5
PPO_EPOCHS   = 4                        # CUDA可以多跑几轮
BATCH_SIZE   = 2500                     # 一次处理全部智能体
UPDATE_EVERY = 100
HISTORY_LEN  = 8
STATE_DIM    = 5
HIDDEN_DIM   = 128                      # 网络稍大

# 实验参数
ALPHA_LIST  = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
B_LIST      = [1.1, 1.2, 1.3, 1.4, 1.5]
B_DEFAULT   = 1.3


# ── 网络 ──
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
        )
        self.actor  = nn.Linear(HIDDEN_DIM, 1)
        self.critic = nn.Linear(HIDDEN_DIM, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h).squeeze(-1), self.critic(h).squeeze(-1)

    def act(self, x):
        logit, value = self(x)
        dist = Bernoulli(logits=logit)
        a = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, x, a):
        logit, value = self(x)
        dist = Bernoulli(logits=logit)
        return dist.log_prob(a), value, dist.entropy()


# ── 环境（纯numpy，快）──
class SpatialPDEnv:
    def __init__(self, N=GRID_N, b=1.3):
        self.N, self.n, self.b = N, N*N, b
        self.max_pay = b * 4
        idx   = np.arange(N*N).reshape(N, N)
        up    = np.roll(idx, -1, axis=0).ravel()
        down  = np.roll(idx,  1, axis=0).ravel()
        left  = np.roll(idx, -1, axis=1).ravel()
        right = np.roll(idx,  1, axis=1).ravel()
        self.nb = np.stack([up, down, left, right], axis=1)  # (n,4)

    def reset(self):
        self.s    = (np.random.rand(self.n) < INIT_COOP).astype(int)
        self.pay  = np.zeros(self.n)
        self.hist = np.zeros((self.n, HISTORY_LEN))
        return self._obs()

    def _obs(self, t=0):
        nb_coop  = self.s[self.nb].sum(1).astype(float)
        c_ratio  = nb_coop / 4.0
        own_s    = self.s.astype(float)
        own_pay  = np.clip(self.pay / self.max_pay, 0, 1)
        hist_avg = np.clip(self.hist.mean(1) / self.max_pay, 0, 1)
        t_norm   = min(float(t) / T_STEPS, 1.0)
        return np.stack([
            c_ratio, own_s, own_pay, hist_avg,
            np.full(self.n, t_norm)
        ], axis=1).astype(np.float32)

    def step(self, actions, alpha, t=0):
        self.s   = actions.copy()
        nb_coop  = self.s[self.nb].sum(1).astype(float)
        self.pay = np.where(self.s == 1, nb_coop, self.b * nb_coop)

        coop_rate = self.s.mean()
        ind_r  = self.pay / self.max_pay
        reward = alpha * ind_r + (1.0 - alpha) * coop_rate

        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = self.pay

        return self._obs(t), reward.astype(np.float32), coop_rate


# ── PPO更新（CUDA优化：pin_memory + non_blocking）──
def ppo_update(net, opt, s_buf, a_buf, lp_buf, r_buf, v_buf):
    if not s_buf: return

    S  = torch.tensor(np.concatenate(s_buf),  dtype=torch.float32).to(DEVICE, non_blocking=True)
    A  = torch.tensor(np.concatenate(a_buf),  dtype=torch.float32).to(DEVICE, non_blocking=True)
    LP = torch.tensor(np.concatenate(lp_buf), dtype=torch.float32).to(DEVICE, non_blocking=True)
    R  = torch.tensor(np.concatenate(r_buf),  dtype=torch.float32).to(DEVICE, non_blocking=True)
    V  = torch.tensor(np.concatenate(v_buf),  dtype=torch.float32).to(DEVICE, non_blocking=True)

    R   = (R - R.mean()) / (R.std() + 1e-8)
    adv = R - V.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = S.shape[0]
    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            mb  = idx[i:i+BATCH_SIZE]
            nlp, nv, ent = net.evaluate(S[mb], A[mb])
            ratio = torch.exp(nlp - LP[mb])
            loss  = (
                -torch.min(ratio * adv[mb],
                           torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv[mb]).mean()
                + VALUE_COEF  * (nv - R[mb]).pow(2).mean()
                - ENTROPY_COEF * ent.mean()
            )
            opt.zero_grad(set_to_none=True)   # CUDA优化：set_to_none更快
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


# ── 单次仿真 ──
def run_once(b, alpha, seed, verbose=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SpatialPDEnv(b=b)
    net = ActorCritic().to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    obs = env.reset()
    s_buf, a_buf, lp_buf, r_buf, v_buf = [], [], [], [], []
    coop_hist = []

    for t in range(T_STEPS):
        st = torch.tensor(obs, dtype=torch.float32).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            a, lp, v = net.act(st)
        a_np = a.cpu().numpy().astype(int)

        obs2, r, cr = env.step(a_np, alpha, t)
        coop_hist.append(cr)

        s_buf.append(obs)
        a_buf.append(a_np.astype(np.float32))
        lp_buf.append(lp.cpu().numpy())
        r_buf.append(r)
        v_buf.append(v.cpu().numpy())
        obs = obs2

        if (t+1) % UPDATE_EVERY == 0:
            ppo_update(net, opt, s_buf, a_buf, lp_buf, r_buf, v_buf)
            s_buf.clear(); a_buf.clear(); lp_buf.clear()
            r_buf.clear(); v_buf.clear()

        if verbose and (t+1) % 100 == 0:
            print(f"    t={t+1:4d}  coop={cr:.3f}", flush=True)

    return np.array(coop_hist)


# ── 实验1：sweep α，固定b ──
def exp_sweep_alpha(b=B_DEFAULT, save_dir='results'):
    print(f"\n{'='*55}")
    print(f"Exp1: Sweep α | b={b} | seeds={N_SEEDS}")
    print(f"{'='*55}")
    os.makedirs(save_dir, exist_ok=True)

    all_hists  = {}
    final_mean = {}
    final_std  = {}

    total = len(ALPHA_LIST) * N_SEEDS
    count = 0

    for alpha in ALPHA_LIST:
        runs = []
        for seed in range(N_SEEDS):
            t0   = time.time()
            hist = run_once(b=b, alpha=alpha, seed=seed, verbose=True)
            runs.append(hist)
            count += 1
            print(f"  [{count:3d}/{total}] α={alpha:.2f} seed={seed} "
                  f"final={hist[-300:].mean():.3f}  {time.time()-t0:.1f}s")
        hists = np.array(runs)
        all_hists[alpha]  = hists
        final_mean[alpha] = hists[:, -300:].mean()
        final_std[alpha]  = hists[:, -300:].mean(1).std()

    np.save(f'{save_dir}/exp1_alpha_b{b}.npy',
            {'hists': all_hists, 'mean': final_mean, 'std': final_std},
            allow_pickle=True)

    _plot_alpha(all_hists, final_mean, final_std, b, save_dir)

    # 打印汇总表
    print(f"\n{'─'*40}")
    print(f"  {'α':>5}  {'合作率(mean)':>12}  {'std':>6}")
    for alpha in ALPHA_LIST:
        marker = ' ◀' if 0.3 < final_mean[alpha] < 0.7 else ''
        print(f"  {alpha:.1f}    {final_mean[alpha]:.3f}         "
              f"{final_std[alpha]:.3f}{marker}")

    return all_hists, final_mean, final_std


# ── 实验2：sweep b，多个α对比 ──
def exp_sweep_b(alpha_compare=None, save_dir='results'):
    if alpha_compare is None:
        alpha_compare = [0.0, 0.3, 0.6, 1.0]
    print(f"\n{'='*55}")
    print(f"Exp2: Sweep b | α={alpha_compare} | seeds={N_SEEDS}")
    print(f"{'='*55}")
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    total = len(alpha_compare) * len(B_LIST) * N_SEEDS
    count = 0

    for alpha in alpha_compare:
        for b in B_LIST:
            runs = []
            for seed in range(N_SEEDS):
                t0   = time.time()
                hist = run_once(b=b, alpha=alpha, seed=seed, verbose=True)
                runs.append(hist[-300:].mean())
                count += 1
                print(f"  [{count:3d}/{total}] α={alpha:.2f} b={b:.1f} "
                      f"seed={seed} final={runs[-1]:.3f}  {time.time()-t0:.1f}s")
            results[(alpha, b)] = (np.mean(runs), np.std(runs))

    np.save(f'{save_dir}/exp2_sweep_b.npy', results, allow_pickle=True)
    _plot_b(results, alpha_compare, save_dir)
    return results


# ── 绘图：实验1 ──
def _plot_alpha(all_hists, final_mean, final_std, b, save_dir):
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f'PPO with Mixed Reward  (b={b}, {GRID_N}×{GRID_N}, von Neumann)',
        fontweight='bold')

    cm = plt.cm.RdYlGn(np.linspace(0.05, 0.95, len(ALPHA_LIST)))

    # (a) 时序
    ax = axes[0]
    window = 100
    for alpha, col in zip(ALPHA_LIST, cm):
        h = all_hists[alpha]
        m = h.mean(0)
        smooth = np.convolve(m, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(m)), smooth,
                color=col, lw=2, label=f'α={alpha:.2f}')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time Steps', ylabel='Cooperation ρ_C', ylim=(0, 1.05))
    ax.set_title('(a) Temporal Evolution', fontweight='bold')
    ax.legend(ncol=2, fontsize=7.5); ax.grid(alpha=0.25)

    # (b) 临界图
    ax = axes[1]
    alphas = np.array(ALPHA_LIST)
    means  = np.array([final_mean[a] for a in ALPHA_LIST])
    stds   = np.array([final_std[a]  for a in ALPHA_LIST])
    ax.errorbar(alphas, means, yerr=stds, fmt='o-', color='steelblue',
                capsize=5, lw=2, markersize=6)
    ax.fill_between(alphas, means-stds, means+stds, alpha=0.15, color='steelblue')
    ax.axhline(0.5, color='gray', ls='--', lw=1)

    # 标注临界点
    for k in range(len(means)-1):
        if means[k] > 0.5 >= means[k+1] or means[k] < 0.5 <= means[k+1]:
            alpha_crit = (alphas[k] + alphas[k+1]) / 2
            ax.axvline(alpha_crit, color='red', ls=':', lw=2,
                       label=f'α*≈{alpha_crit:.2f}')
            print(f"\n  >> 临界点 α* ≈ {alpha_crit:.2f}")
            break

    ax.set(xlabel='Self-interest Weight α',
           ylabel='Final Cooperation ρ_C',
           ylim=(0, 1.05), xticks=alphas)
    ax.set_title('(b) Critical Threshold α*', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = f'{save_dir}/exp1_alpha_b{b}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# ── 绘图：实验2 ──
def _plot_b(results, alpha_compare, save_dir):
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, len(alpha_compare)))

    for alpha, col in zip(alpha_compare, colors):
        bs = np.array(B_LIST)
        m  = np.array([results[(alpha, b)][0] for b in bs])
        s  = np.array([results[(alpha, b)][1] for b in bs])
        ax.errorbar(bs, m, yerr=s, fmt='o-', color=col,
                    capsize=4, lw=2, markersize=5, label=f'α={alpha:.2f}')
        ax.fill_between(bs, m-s, m+s, color=col, alpha=0.1)

    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Defection Temptation b',
           ylabel='Final Cooperation ρ_C',
           ylim=(0, 1.05))
    ax.set_title(f'Cooperation vs Temptation  (PPO, {GRID_N}×{GRID_N})',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    path = f'{save_dir}/exp2_sweep_b.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# ── 主程序 ──
if __name__ == '__main__':
    SAVE_DIR = 'results'
    t0_total = time.time()

    print(f"设备: {DEVICE}")
    print(f"格子: {GRID_N}×{GRID_N} | 步数: {T_STEPS} | 种子: {N_SEEDS}")

    # 只跑实验1：细扫α，固定b=1.3，单种子
    exp_sweep_alpha(b=B_DEFAULT, save_dir=SAVE_DIR)

    print(f"\nAll done.  Total: {(time.time()-t0_total)/60:.1f} min")