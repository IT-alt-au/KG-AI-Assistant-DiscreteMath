"""
Meta-Imitation PPO for Spatial Prisoner's Dilemma
==================================================
创新点：PPO作为元策略，控制"何时执行Fermi模仿"

动作空间：
  a=1 → 执行Fermi模仿（以Fermi概率复制收益更高邻居的策略）
  a=0 → 保持当前策略不变（自主决策）

PPO奖励：
  执行模仿且收益提升 → 正奖励
  执行模仿且收益下降 → 负奖励
  保持策略且收益稳定/提升 → 正奖励

State (dim=6):
  0: 邻居合作比例
  1: 自身当前策略 (1=C, 0=D)
  2: 自身当前收益（归一化）
  3: 最佳邻居收益 - 自身收益（归一化，Fermi信号）
  4: 自身历史平均收益（归一化）
  5: 时间归一化

对比基线：纯Fermi（无PPO控制）
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

# ── 设备 ──
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); print("✓ MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda"); print("✓ CUDA")
else:
    DEVICE = torch.device("cpu");  print("⚠ CPU")

# ── 超参数 ──
GRID_N       = 50
N_AGENTS     = GRID_N * GRID_N
T_STEPS      = 5000
FERMI_K      = 0.1       # Fermi温度（理性程度，越小越理性）
INIT_COOP    = 0.5

LR           = 3e-4
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF   = 0.5
PPO_EPOCHS   = 3
BATCH_SIZE   = 1024
UPDATE_EVERY = 100
HISTORY_LEN  = 8
STATE_DIM    = 6
HIDDEN_DIM   = 64


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
        a    = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, x, a):
        logit, value = self(x)
        dist = Bernoulli(logits=logit)
        return dist.log_prob(a), value, dist.entropy()


# ── 环境 ──
class SpatialPDEnv:
    def __init__(self, N=GRID_N, b=1.3):
        self.N, self.n, self.b = N, N*N, b
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
        return self._obs(t=0)

    def _payoffs(self, s=None):
        if s is None: s = self.s
        nb_coop = s[self.nb].sum(1).astype(float)
        return np.where(s == 1, nb_coop, self.b * nb_coop)

    def _obs(self, t=0):
        nb_s     = self.s[self.nb]
        nb_coop  = nb_s.sum(1).astype(float)
        nb_pay   = self.pay[self.nb]            # (n,4)
        best_nb  = nb_pay.max(1)                # 最佳邻居收益

        max_pay  = self.b * 4 + 1e-8
        c_ratio  = nb_coop / 4.0
        own_s    = self.s.astype(float)
        own_pay  = np.clip(self.pay / max_pay, 0, 1)
        fermi_sig= np.clip((best_nb - self.pay) / max_pay, -1, 1)  # Fermi信号
        hist_avg = np.clip(self.hist.mean(1) / max_pay, 0, 1)
        t_norm   = min(float(t) / T_STEPS, 1.0)

        return np.stack([
            c_ratio, own_s, own_pay,
            fermi_sig, hist_avg,
            np.full(self.n, t_norm)
        ], axis=1).astype(np.float32)

    def step(self, do_imitate, t=0):
        """
        do_imitate: (n,) int  1=执行Fermi模仿, 0=保持不变
        Fermi模仿：随机选一个邻居，以Fermi概率复制其策略
        """
        old_pay = self.pay.copy()
        new_s   = self.s.copy()

        # 对选择模仿的智能体执行Fermi更新
        imit_idx = np.where(do_imitate == 1)[0]
        if len(imit_idx) > 0:
            # 随机选一个邻居
            nb_choice = self.nb[imit_idx, np.random.randint(0, 4, len(imit_idx))]
            pay_self  = self.pay[imit_idx]
            pay_nb    = self.pay[nb_choice]
            # Fermi概率
            max_p = self.b * 4 + 1e-8
            fermi_prob = 1.0 / (1.0 + np.exp((pay_self - pay_nb) / (FERMI_K * max_p)))
            do_copy    = np.random.rand(len(imit_idx)) < fermi_prob
            new_s[imit_idx[do_copy]] = self.s[nb_choice[do_copy]]

        self.s   = new_s
        self.pay = self._payoffs()

        # ── 奖励设计 ──
        pay_change = self.pay - old_pay                    # 收益变化
        max_pay    = self.b * 4 + 1e-8

        # 执行模仿的人：收益提升给正奖励，下降给负奖励
        # 保持策略的人：收益稳定/提升给正奖励
        reward = pay_change / max_pay                      # 基础：收益变化

        # 额外：模仿且合作比例提升时有bonus
        nb_coop_ratio = self.s[self.nb].sum(1) / 4.0
        coop_bonus    = 0.1 * nb_coop_ratio * self.s      # 合作且邻居多合作
        reward += coop_bonus

        # 更新历史
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = self.pay

        return self._obs(t), reward.astype(np.float32), self.s.mean()


# ── PPO更新 ──
def ppo_update(net, opt, s_buf, a_buf, lp_buf, r_buf, v_buf):
    if not s_buf: return
    S  = torch.tensor(np.concatenate(s_buf),  dtype=torch.float32).to(DEVICE)
    A  = torch.tensor(np.concatenate(a_buf),  dtype=torch.float32).to(DEVICE)
    LP = torch.tensor(np.concatenate(lp_buf), dtype=torch.float32).to(DEVICE)
    R  = torch.tensor(np.concatenate(r_buf),  dtype=torch.float32).to(DEVICE)
    V  = torch.tensor(np.concatenate(v_buf),  dtype=torch.float32).to(DEVICE)

    R   = (R - R.mean()) / (R.std() + 1e-8)
    adv = R - V.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = S.shape[0]
    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            mb = idx[i:i+BATCH_SIZE]
            nlp, nv, ent = net.evaluate(S[mb], A[mb])
            ratio = torch.exp(nlp - LP[mb])
            loss  = (
                -torch.min(ratio*adv[mb],
                           torch.clamp(ratio,1-CLIP_EPS,1+CLIP_EPS)*adv[mb]).mean()
                + VALUE_COEF * (nv - R[mb]).pow(2).mean()
                - ENTROPY_COEF * ent.mean()
            )
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


# ── Meta-Imitation PPO ──
def run_meta_ppo(b=1.3, seed=0, verbose=True):
    np.random.seed(seed); torch.manual_seed(seed)
    env = SpatialPDEnv(b=b)
    net = ActorCritic().to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    obs = env.reset()
    s_buf, a_buf, lp_buf, r_buf, v_buf = [], [], [], [], []
    coop_hist  = []
    imit_hist  = []   # 每步选择模仿的比例

    for t in range(T_STEPS):
        st = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            a, lp, v = net.act(st)
        a_np = a.cpu().numpy().astype(int)

        obs2, r, cr = env.step(a_np, t)
        coop_hist.append(cr)
        imit_hist.append(a_np.mean())

        s_buf.append(obs); a_buf.append(a_np.astype(np.float32))
        lp_buf.append(lp.cpu().numpy()); r_buf.append(r)
        v_buf.append(v.cpu().numpy())
        obs = obs2

        if (t+1) % UPDATE_EVERY == 0:
            ppo_update(net, opt, s_buf, a_buf, lp_buf, r_buf, v_buf)
            s_buf.clear(); a_buf.clear(); lp_buf.clear()
            r_buf.clear(); v_buf.clear()

        if verbose and (t+1) % 500 == 0:
            print(f"  t={t+1:4d}  coop={cr:.3f}  imit_rate={a_np.mean():.2f}")

    return np.array(coop_hist), np.array(imit_hist)


# ── 纯Fermi基线 ──
def run_pure_fermi(b=1.3, seed=0):
    """所有人每步都执行Fermi模仿，作为对比基线"""
    np.random.seed(seed)
    env  = SpatialPDEnv(b=b)
    env.reset()
    coop_hist = []

    for t in range(T_STEPS):
        # 全部执行Fermi
        all_imitate = np.ones(N_AGENTS, dtype=int)
        _, _, cr = env.step(all_imitate, t)
        coop_hist.append(cr)

    return np.array(coop_hist)


# ── 主程序 ──
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    b = 1.3

    print(f"\n{'='*50}")
    print(f"Meta-Imitation PPO  b={b}  T={T_STEPS}")
    print(f"{'='*50}")

    t0 = time.time()
    print("\n[1] Meta-Imitation PPO:")
    ch_ppo, ch_imit = run_meta_ppo(b=b, seed=0, verbose=True)

    print("\n[2] 纯Fermi基线:")
    ch_fermi = run_pure_fermi(b=b, seed=0)
    print(f"  Fermi最终合作率: {ch_fermi[-500:].mean():.3f}")

    print(f"\n{'='*40}")
    print(f"Meta-PPO 最终合作率  (last 500): {ch_ppo[-500:].mean():.3f}")
    print(f"Fermi    最终合作率  (last 500): {ch_fermi[-500:].mean():.3f}")
    print(f"Meta-PPO 平均模仿率  (last 500): {ch_imit[-500:].mean():.3f}")
    print(f"耗时: {(time.time()-t0)/60:.1f} min")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Meta-Imitation PPO vs Pure Fermi  (b={b}, {GRID_N}×{GRID_N})',
                 fontweight='bold')

    ax = axes[0]
    window = 100
    for data, color, label in [
        (ch_ppo,   '#2ca02c', 'Meta-Imitation PPO'),
        (ch_fermi, '#d62728', 'Pure Fermi'),
    ]:
        ax.plot(data, color=color, lw=1, alpha=0.4)
        smooth = np.convolve(data, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(data)), smooth, color=color, lw=2.5, label=label)
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time Steps', ylabel='Cooperation ρ_C', ylim=(0,1.05))
    ax.set_title('(a) Cooperation Rate')
    ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1]
    smooth_imit = np.convolve(ch_imit, np.ones(window)/window, mode='valid')
    ax.plot(ch_imit, color='#ff7f0e', lw=1, alpha=0.4)
    ax.plot(range(window-1, len(ch_imit)), smooth_imit,
            color='#ff7f0e', lw=2.5, label='Imitation rate')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time Steps', ylabel='Fraction choosing to imitate',
           ylim=(0,1.05))
    ax.set_title('(b) PPO Imitation Decision')
    ax.legend(); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('results/meta_imit_ppo_b1.3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图已保存: results/meta_imit_ppo_b1.3.png")