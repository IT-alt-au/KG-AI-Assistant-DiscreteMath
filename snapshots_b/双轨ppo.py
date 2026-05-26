"""
Spatial Prisoner's Dilemma — Dual-Track PPO
============================================
创新点：双轨PPO框架
  - 网络A（CoopNet）：合作者使用，学习"何时维持合作"
    奖励 = 合作实际收益 - 假设背叛的反事实收益（机会成本）
    → 正值表示"合作比背叛更划算"

  - 网络B（DefNet）：背叛者使用，学习"何时转向合作"
    奖励 = 合作邻居比例带来的潜在收益提升
    → 正值表示"周围合作者多，转向合作有利"

每步根据智能体当前策略实时分配网络：
  当前=C → CoopNet决策 → 输出维持C或转向D
  当前=D → DefNet决策  → 输出维持D或转向C

对比组：
  - 单网络PPO（原始版本，合作率≈0.11）
  - 双轨PPO（本文方法）

研究问题：
  1. 双轨PPO能否打破背叛均衡？
  2. 两个网络的收敛速度是否不同？
  3. 背叛诱惑b对两个网络的影响是否存在差异？

网格：50×50，von Neumann邻域，周期边界
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

# ══════════════════════════════════════════════
# 设备检测
# ══════════════════════════════════════════════
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Apple Silicon MPS 已启用")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ CUDA 已启用")
else:
    DEVICE = torch.device("cpu")
    print("⚠ 使用 CPU")

# ══════════════════════════════════════════════
# 超参数
# ══════════════════════════════════════════════
GRID_N        = 50
N_AGENTS      = GRID_N * GRID_N        # 2500
T_STEPS       = 2000
N_SEEDS       = 3

B_LIST        = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# PPO超参数
LR_COOP       = 3e-4       # 合作网络学习率
LR_DEF        = 3e-4       # 背叛网络学习率
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.05       # 较大熵系数，保持探索
VALUE_COEF    = 0.5
PPO_EPOCHS    = 3
BATCH_SIZE    = 1024
UPDATE_EVERY  = 100

INIT_COOP     = 0.5        # 初始合作率50%（公平起点）
HISTORY_LEN   = 8

STATE_DIM     = 6          # 见_state()说明
HIDDEN_DIM    = 64


# ══════════════════════════════════════════════
# Actor-Critic 网络（CoopNet和DefNet结构相同，参数独立）
# ══════════════════════════════════════════════
class ActorCritic(nn.Module):
    def __init__(self, name="net"):
        super().__init__()
        self.name = name
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
        )
        self.actor  = nn.Linear(HIDDEN_DIM, 1)  # logit → P(cooperate)
        self.critic = nn.Linear(HIDDEN_DIM, 1)  # V(s)

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
        dist   = Bernoulli(logits=logit)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, x, action):
        logit, value = self(x)
        dist = Bernoulli(logits=logit)
        return dist.log_prob(action), value, dist.entropy()


# ══════════════════════════════════════════════
# 环境
# ══════════════════════════════════════════════
class SpatialPDEnv:
    """
    von Neumann 50×50 空间囚徒困境
    收益矩阵（归一化）：R=1, T=b, S=P=0
    """
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
        return self._state()

    def _compute_payoffs(self, strategies=None):
        """计算给定策略下的收益，默认用当前策略"""
        s = strategies if strategies is not None else self.s
        nb_coop = s[self.nb].sum(1).astype(float)
        return np.where(s == 1, nb_coop, self.b * nb_coop)

    def _state(self, t=0):
        """
        State (dim=6):
          0: 邻居合作比例（真实，全观测）
          1: 邻居背叛比例
          2: 自身历史平均收益（归一化）
          3: 自身当前策略 (1=C, 0=D)
          4: 反事实收益差（假设切换策略后的收益变化，归一化）
          5: 时间归一化
        """
        nb_s     = self.s[self.nb]                        # (n,4)
        nb_coop  = nb_s.sum(1).astype(float)              # (n,)

        c_ratio  = nb_coop / 4.0                          # 邻居合作比
        d_ratio  = 1.0 - c_ratio                          # 邻居背叛比
        own_avg  = np.clip(self.hist.mean(1) / (self.b * 4 + 1e-8), 0, 1)
        own_s    = self.s.astype(float)

        # 反事实收益差：如果切换策略，收益变化多少
        # 当前合作者：切换到背叛的收益增量 = b*nb_coop - nb_coop = (b-1)*nb_coop
        # 当前背叛者：切换到合作的收益增量 = nb_coop - b*nb_coop = (1-b)*nb_coop
        curr_pay  = self._compute_payoffs()
        flip_s    = 1 - self.s
        flip_pay  = np.where(flip_s == 1, nb_coop, self.b * nb_coop)
        cf_diff   = np.clip((flip_pay - curr_pay) / (self.b * 4 + 1e-8), -1, 1)

        t_norm = min(float(t) / T_STEPS, 1.0)

        return np.stack([
            c_ratio, d_ratio, own_avg,
            own_s, cf_diff,
            np.full(self.n, t_norm)
        ], axis=1).astype(np.float32)

    def step(self, actions, t=0):
        """
        actions: (n,) int  1=C, 0=D
        返回：
          next_state
          reward_coop: (n,) CoopNet的奖励（合作者专用）
          reward_def:  (n,) DefNet的奖励（背叛者专用）
          coop_rate
        """
        old_s   = self.s.copy()
        self.s  = actions.copy()
        self.pay = self._compute_payoffs()

        nb_coop = self.s[self.nb].sum(1).astype(float)   # (n,) 更新后邻居合作数

        # ── CoopNet奖励：合作者的机会成本 ──
        # 合作实际收益 - 假设背叛能得到的收益
        # 正值 → 合作比背叛更好 → 强化维持合作
        pay_if_coop = nb_coop                             # 合作时得到的
        pay_if_def  = self.b * nb_coop                   # 背叛时得到的
        reward_coop = (pay_if_coop - pay_if_def) / (self.b * 4 + 1e-8)
        # 注：这对合作者总是负的（因为背叛短期收益更高）
        # 但合作集群里nb_coop大，差值绝对值小 → 相对损失小 → 网络学会在合作环境里维持合作

        # ── DefNet奖励：背叛者的转换激励 ──
        # 奖励 = 邻居合作比例 × (1 - b的惩罚因子)
        # 邻居合作者越多，转向合作的潜在收益越高
        nb_coop_ratio = nb_coop / 4.0
        reward_def    = nb_coop_ratio - 0.5              # 超过50%合作邻居才是正奖励

        # 更新历史
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = self.pay

        next_state = self._state(t)
        coop_rate  = self.s.mean()

        return (next_state,
                reward_coop.astype(np.float32),
                reward_def.astype(np.float32),
                coop_rate)


# ══════════════════════════════════════════════
# PPO更新（通用，传入对应网络和buffer）
# ══════════════════════════════════════════════
def ppo_update(net, opt, s_buf, a_buf, lp_buf, r_buf, v_buf):
    if len(s_buf) == 0:
        return
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
            mb = idx[i:i + BATCH_SIZE]
            nlp, nv, ent = net.evaluate(S[mb], A[mb])
            ratio = torch.exp(nlp - LP[mb])
            loss  = (
                -torch.min(ratio * adv[mb],
                           torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv[mb]).mean()
                + VALUE_COEF  * (nv - R[mb]).pow(2).mean()
                - ENTROPY_COEF * ent.mean()
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


# ══════════════════════════════════════════════
# 单次仿真（双轨PPO）
# ══════════════════════════════════════════════
def run_dual_ppo(b, seed, verbose=False):
    """
    双轨PPO仿真
    返回：
      coop_hist     (T,)   合作率时间序列
      coop_net_hist (T,)   合作者中维持合作的比例
      def_net_hist  (T,)   背叛者中转向合作的比例
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = SpatialPDEnv(b=b)
    coop_net = ActorCritic("CoopNet").to(DEVICE)
    def_net  = ActorCritic("DefNet").to(DEVICE)
    opt_c    = optim.Adam(coop_net.parameters(), lr=LR_COOP, eps=1e-5)
    opt_d    = optim.Adam(def_net.parameters(),  lr=LR_DEF,  eps=1e-5)

    states = env.reset()

    # 独立buffer
    c_s, c_a, c_lp, c_r, c_v = [], [], [], [], []   # CoopNet buffer
    d_s, d_a, d_lp, d_r, d_v = [], [], [], [], []   # DefNet buffer

    coop_hist     = []
    coop_net_hist = []   # 合作者中维持合作的比例
    def_net_hist  = []   # 背叛者中转向合作的比例

    for t in range(T_STEPS):
        st = torch.tensor(states, dtype=torch.float32).to(DEVICE)

        # 分流：当前策略决定用哪个网络
        is_coop = (env.s == 1)                            # (n,) bool
        is_def  = ~is_coop

        actions  = np.zeros(N_AGENTS, dtype=int)
        lp_store = np.zeros(N_AGENTS, dtype=np.float32)
        v_store  = np.zeros(N_AGENTS, dtype=np.float32)

        # ── CoopNet处理合作者 ──
        coop_idx = np.where(is_coop)[0]
        if len(coop_idx) > 0:
            st_c = st[coop_idx]
            with torch.no_grad():
                a_c, lp_c, v_c = coop_net.act(st_c)
            actions[coop_idx]  = a_c.cpu().numpy().astype(int)
            lp_store[coop_idx] = lp_c.cpu().numpy()
            v_store[coop_idx]  = v_c.cpu().numpy()

        # ── DefNet处理背叛者 ──
        def_idx = np.where(is_def)[0]
        if len(def_idx) > 0:
            st_d = st[def_idx]
            with torch.no_grad():
                a_d, lp_d, v_d = def_net.act(st_d)
            actions[def_idx]  = a_d.cpu().numpy().astype(int)
            lp_store[def_idx] = lp_d.cpu().numpy()
            v_store[def_idx]  = v_d.cpu().numpy()

        # 环境步进
        ns, r_coop, r_def, cr = env.step(actions, t)

        # 记录统计
        coop_hist.append(cr)
        if len(coop_idx) > 0:
            coop_net_hist.append(actions[coop_idx].mean())   # 合作者中选C的比例
        else:
            coop_net_hist.append(0.0)
        if len(def_idx) > 0:
            def_net_hist.append(actions[def_idx].mean())     # 背叛者中选C的比例
        else:
            def_net_hist.append(0.0)

        # 存入各自buffer
        if len(coop_idx) > 0:
            c_s.append(states[coop_idx])
            c_a.append(actions[coop_idx].astype(np.float32))
            c_lp.append(lp_store[coop_idx])
            c_r.append(r_coop[coop_idx])
            c_v.append(v_store[coop_idx])

        if len(def_idx) > 0:
            d_s.append(states[def_idx])
            d_a.append(actions[def_idx].astype(np.float32))
            d_lp.append(lp_store[def_idx])
            d_r.append(r_def[def_idx])
            d_v.append(v_store[def_idx])

        states = ns

        # PPO更新
        if (t + 1) % UPDATE_EVERY == 0:
            ppo_update(coop_net, opt_c, c_s, c_a, c_lp, c_r, c_v)
            ppo_update(def_net,  opt_d, d_s, d_a, d_lp, d_r, d_v)
            c_s.clear(); c_a.clear(); c_lp.clear(); c_r.clear(); c_v.clear()
            d_s.clear(); d_a.clear(); d_lp.clear(); d_r.clear(); d_v.clear()

        if verbose and (t + 1) % 500 == 0:
            print(f"    t={t+1}  coop={cr:.3f}  "
                  f"CoopNet维持={coop_net_hist[-1]:.2f}  "
                  f"DefNet转化={def_net_hist[-1]:.2f}")

    return (np.array(coop_hist),
            np.array(coop_net_hist),
            np.array(def_net_hist))


# ══════════════════════════════════════════════
# 单网络PPO基线（对照组）
# ══════════════════════════════════════════════
def run_single_ppo(b, seed):
    """单网络PPO，作为对照组"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SpatialPDEnv(b=b)
    net = ActorCritic("SingleNet").to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    states = env.reset()
    s_b, a_b, lp_b, r_b, v_b = [], [], [], [], []
    coop_hist = []

    for t in range(T_STEPS):
        st = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            a, lp, v = net.act(st)
        a_np = a.cpu().numpy().astype(int)

        ns, r_c, r_d, cr = env.step(a_np, t)
        # 单网络用平均奖励
        r_avg = (r_c + r_d) / 2.0

        coop_hist.append(cr)
        s_b.append(states); a_b.append(a_np.astype(np.float32))
        lp_b.append(lp.cpu().numpy()); r_b.append(r_avg); v_b.append(v.cpu().numpy())
        states = ns

        if (t + 1) % UPDATE_EVERY == 0:
            ppo_update(net, opt, s_b, a_b, lp_b, r_b, v_b)
            s_b.clear(); a_b.clear(); lp_b.clear(); r_b.clear(); v_b.clear()

    return np.array(coop_hist)


# ══════════════════════════════════════════════
# 实验1：固定b，对比单/双轨PPO时序
# ══════════════════════════════════════════════
def exp_compare_methods(b=1.3, save_dir='results'):
    print(f"\n{'='*55}")
    print(f"Exp1: 单网络 vs 双轨PPO  b={b}  seeds={N_SEEDS}")
    print(f"{'='*55}")
    os.makedirs(save_dir, exist_ok=True)

    dual_coops, dual_cnet, dual_dnet = [], [], []
    single_coops = []

    for seed in range(N_SEEDS):
        t0 = time.time()
        ch, cn, dn = run_dual_ppo(b=b, seed=seed, verbose=True)
        dual_coops.append(ch); dual_cnet.append(cn); dual_dnet.append(dn)
        print(f"  [Dual ] seed={seed} final={ch[-300:].mean():.3f}  {time.time()-t0:.1f}s")

        t0 = time.time()
        sh = run_single_ppo(b=b, seed=seed)
        single_coops.append(sh)
        print(f"  [Single] seed={seed} final={sh[-300:].mean():.3f}  {time.time()-t0:.1f}s")

    results = {
        'dual_coop':  np.array(dual_coops),
        'dual_cnet':  np.array(dual_cnet),
        'dual_dnet':  np.array(dual_dnet),
        'single_coop': np.array(single_coops),
    }
    np.save(f'{save_dir}/exp1_compare_b{b}.npy', results, allow_pickle=True)
    _plot_compare(results, b, save_dir)
    return results


# ══════════════════════════════════════════════
# 实验2：扫描b，对比两种方法
# ══════════════════════════════════════════════
def exp_sweep_b(save_dir='results'):
    print(f"\n{'='*55}")
    print(f"Exp2: Sweep b  seeds={N_SEEDS}")
    print(f"{'='*55}")
    os.makedirs(save_dir, exist_ok=True)

    dual_res   = {}   # b -> (mean, std)
    single_res = {}

    total = len(B_LIST) * N_SEEDS
    count = 0

    for b in B_LIST:
        d_runs, s_runs = [], []
        for seed in range(N_SEEDS):
            t0 = time.time()
            ch, _, _ = run_dual_ppo(b=b, seed=seed)
            sh        = run_single_ppo(b=b, seed=seed)
            d_runs.append(ch[-300:].mean())
            s_runs.append(sh[-300:].mean())
            count += 1
            print(f"  [{count:2d}/{total}] b={b:.1f} seed={seed}  "
                  f"Dual={d_runs[-1]:.3f}  Single={s_runs[-1]:.3f}  "
                  f"{time.time()-t0:.1f}s")
        dual_res[b]   = (np.mean(d_runs), np.std(d_runs))
        single_res[b] = (np.mean(s_runs), np.std(s_runs))

    np.save(f'{save_dir}/exp2_sweep_b.npy',
            {'dual': dual_res, 'single': single_res}, allow_pickle=True)
    _plot_sweep_b(dual_res, single_res, save_dir)
    return dual_res, single_res


# ══════════════════════════════════════════════
# 绘图：实验1
# ══════════════════════════════════════════════
def _plot_compare(results, b, save_dir):
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f'Dual-Track PPO vs Single-Network PPO  '
        f'(von Neumann {GRID_N}×{GRID_N}, b={b})',
        fontsize=11, fontweight='bold')

    def _plot_band(ax, data, color, label):
        m = data.mean(0); s = data.std(0)
        t = np.arange(len(m))
        ax.plot(t, m, color=color, lw=2, label=label)
        ax.fill_between(t, m-s, m+s, color=color, alpha=0.15)

    # (a) 总合作率对比
    ax = axes[0]
    _plot_band(ax, results['dual_coop'],   '#2ca02c', 'Dual-Track PPO')
    _plot_band(ax, results['single_coop'], '#d62728', 'Single-Network PPO')
    ax.set(xlabel='Time Steps', ylabel='Cooperation ρ_C', ylim=(0, 1.05))
    ax.set_title('(a) Overall Cooperation', y=-0.18, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (b) CoopNet：合作者中维持合作的比例
    ax = axes[1]
    _plot_band(ax, results['dual_cnet'], '#1f77b4', 'CoopNet (C→C rate)')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time Steps', ylabel='Rate of Staying Cooperative',
           ylim=(0, 1.05))
    ax.set_title('(b) CoopNet: Maintaining Cooperation', y=-0.18, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (c) DefNet：背叛者中转向合作的比例
    ax = axes[2]
    _plot_band(ax, results['dual_dnet'], '#ff7f0e', 'DefNet (D→C rate)')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time Steps', ylabel='Rate of Converting to Cooperation',
           ylim=(0, 1.05))
    ax.set_title('(c) DefNet: Converting Defectors', y=-0.18, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.subplots_adjust(bottom=0.2, wspace=0.35)
    path = f'{save_dir}/exp1_compare_b{b}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# ══════════════════════════════════════════════
# 绘图：实验2
# ══════════════════════════════════════════════
def _plot_sweep_b(dual_res, single_res, save_dir):
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bs = np.array(B_LIST)
    for res, color, label in [
        (dual_res,   '#2ca02c', 'Dual-Track PPO'),
        (single_res, '#d62728', 'Single-Network PPO'),
    ]:
        m = np.array([res[b][0] for b in bs])
        s = np.array([res[b][1] for b in bs])
        ax.errorbar(bs, m, yerr=s, fmt='o-', color=color,
                    capsize=4, lw=2, markersize=5, label=label)
        ax.fill_between(bs, m-s, m+s, color=color, alpha=0.1)

    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Defection Temptation b',
           ylabel='Final Cooperation Rate ρ_C',
           ylim=(0, 1.05), xticks=B_LIST)
    ax.set_title(f'Dual-Track vs Single PPO: Robustness to Temptation\n'
                 f'(Spatial PD, {GRID_N}×{GRID_N})', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    path = f'{save_dir}/exp2_sweep_b.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════
if __name__ == '__main__':
    SAVE_DIR = 'results'
    t0 = time.time()

    print(f"设备: {DEVICE}")
    print(f"格子: {GRID_N}×{GRID_N} | 步数: {T_STEPS} | 种子: {N_SEEDS}")

    # 实验1：b=1.3下的详细对比（含CoopNet/DefNet分析）
    exp_compare_methods(b=1.3, save_dir=SAVE_DIR)

    # 实验2：扫描b，双轨 vs 单网络
    exp_sweep_b(save_dir=SAVE_DIR)

    print(f"\nAll done.  Total: {(time.time()-t0)/60:.1f} min")