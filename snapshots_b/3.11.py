"""
Spatial Prisoner's Dilemma with PPO under Partial Observability
================================================================
Research Question: Does a critical visibility threshold p* exist
such that cooperation collapses below it and emerges above it?

Grid:     100x100, von Neumann neighborhood (4 neighbors)
Decision: PPO (partial obs) vs Fermi imitation baseline
State:    [visible_coop_ratio, visible_defect_ratio, own_avg_reward, obs_count_ratio]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import os
import time

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
GRID_N       = 100       # lattice size
N_AGENTS     = GRID_N ** 2
T_STEPS      = 2000      # total time steps
B_RANGE      = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
P_VIS_RANGE  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS      = 3         # independent runs per config
KAPPA        = 0.1       # Fermi noise
EPSILON      = 0.3       # prob of using Fermi imitation (1-eps = PPO)
HISTORY_LEN  = 5         # reward history window

# PPO hyperparams
LR           = 3e-4
GAMMA        = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS   = 4
BATCH_SIZE   = 512

STATE_DIM    = 5         # [coop_ratio_vis, defect_ratio_vis, unobs_ratio, avg_reward_norm, time_norm]
HIDDEN_DIM   = 64

DEVICE = torch.device("cpu")  # CPU is fine for this scale


# ─────────────────────────────────────────────
# PPO Network (shared Actor-Critic)
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden_dim, 1)   # logit for P(cooperate)
        self.critic_head = nn.Linear(hidden_dim, 1)   # value estimate

    def forward(self, x):
        h = self.shared(x)
        logit = self.actor_head(h).squeeze(-1)
        value = self.critic_head(h).squeeze(-1)
        return logit, value

    def get_action_and_logprob(self, x):
        logit, value = self(x)
        dist  = Bernoulli(logits=logit)
        action = dist.sample()                        # 1=cooperate, 0=defect
        logp   = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, x, action):
        logit, value = self(x)
        dist    = Bernoulli(logits=logit)
        logp    = dist.log_prob(action)
        entropy = dist.entropy()
        return logp, value, entropy


# ─────────────────────────────────────────────
# Environment: Spatial PD lattice
# ─────────────────────────────────────────────
class SpatialPDEnv:
    def __init__(self, N=GRID_N, b=1.3, p_vis=1.0):
        self.N     = N
        self.b     = b
        self.p_vis = p_vis          # visibility probability per neighbor
        self.n     = N * N

        # Von Neumann neighbors (up, down, left, right) with periodic boundary
        self._build_neighbor_index()

        # State
        self.strategies  = None     # (n,) int: 1=C, 0=D
        self.payoffs     = None     # (n,) float
        self.reward_hist = None     # (n, HISTORY_LEN) float

    def _build_neighbor_index(self):
        N = self.N
        idx = np.arange(N * N).reshape(N, N)
        up    = np.roll(idx, -1, axis=0).ravel()
        down  = np.roll(idx,  1, axis=0).ravel()
        left  = np.roll(idx, -1, axis=1).ravel()
        right = np.roll(idx,  1, axis=1).ravel()
        # neighbors[i] = [up, down, left, right]
        self.neighbors = np.stack([up, down, left, right], axis=1)  # (n, 4)

    def reset(self):
        self.strategies  = np.random.randint(0, 2, self.n)
        self.payoffs     = np.zeros(self.n)
        self.reward_hist = np.zeros((self.n, HISTORY_LEN))
        return self._get_states()

    def _compute_payoffs(self):
        """Vectorized payoff computation."""
        b = self.b
        s = self.strategies                        # (n,)
        nb_strats = self.strategies[self.neighbors] # (n, 4)
        # payoff: R=1 for C-C, T=b for D-C, S=0 for C-D, P=0 for D-D
        # agent i cooperates: gets sum of neighbors who also cooperate
        # agent i defects:    gets b * sum of neighbors who cooperate
        nb_coop = nb_strats.sum(axis=1).astype(float)  # (n,)
        payoff  = np.where(s == 1, nb_coop, b * nb_coop)
        return payoff

    def _get_states(self, t=0):
        """
        Build state vectors under partial observability.
        Each agent independently samples which neighbors are visible.
        State: [coop_ratio_vis, defect_ratio_vis, unobs_ratio, avg_reward_norm, time_norm]
        """
        n, p = self.n, self.p_vis
        nb_strats = self.strategies[self.neighbors]  # (n, 4)

        # Visibility mask: each neighbor slot independently visible with prob p
        vis_mask = (np.random.rand(n, 4) < p).astype(float)  # (n, 4)
        vis_count = vis_mask.sum(axis=1)                       # (n,)

        vis_coop   = (nb_strats * vis_mask).sum(axis=1)        # (n,)
        vis_defect = ((1 - nb_strats) * vis_mask).sum(axis=1)  # (n,)

        # Ratios (avoid div by zero)
        safe_vis = np.maximum(vis_count, 1)
        coop_ratio   = vis_coop   / safe_vis
        defect_ratio = vis_defect / safe_vis
        unobs_ratio  = (4 - vis_count) / 4.0

        avg_reward = self.reward_hist.mean(axis=1) / 4.0       # normalize by max possible
        time_norm  = min(t / T_STEPS, 1.0)

        states = np.stack([coop_ratio, defect_ratio, unobs_ratio,
                           avg_reward, np.full(n, time_norm)], axis=1).astype(np.float32)
        return states

    def step(self, ppo_actions, use_ppo_mask, t=0):
        """
        ppo_actions:   (n,) int {0,1} from PPO
        use_ppo_mask:  (n,) bool, True = use PPO, False = use Fermi
        """
        old_strats = self.strategies.copy()

        # --- PPO agents update ---
        self.strategies[use_ppo_mask] = ppo_actions[use_ppo_mask]

        # --- Fermi imitation agents update ---
        fermi_idx = np.where(~use_ppo_mask)[0]
        if len(fermi_idx) > 0:
            # Pick random neighbor for each Fermi agent
            nb_choice = np.random.randint(0, 4, len(fermi_idx))
            nb_idx    = self.neighbors[fermi_idx, nb_choice]
            Pi = self.payoffs[fermi_idx]
            Pj = self.payoffs[nb_idx]
            prob = 1.0 / (1.0 + np.exp((Pi - Pj) / KAPPA))
            adopt = np.random.rand(len(fermi_idx)) < prob
            self.strategies[fermi_idx[adopt]] = old_strats[nb_idx[adopt]]

        # --- Compute new payoffs ---
        self.payoffs = self._compute_payoffs()

        # Update reward history (rolling)
        self.reward_hist = np.roll(self.reward_hist, -1, axis=1)
        self.reward_hist[:, -1] = self.payoffs

        next_states = self._get_states(t)
        coop_rate   = self.strategies.mean()
        return next_states, self.payoffs.copy(), coop_rate


# ─────────────────────────────────────────────
# PPO Rollout Buffer
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.states   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []
        self.values   = []
        self.dones    = []

    def clear(self):
        self.__init__()

    def add(self, s, a, lp, r, v):
        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.values.append(v)

    def compute_returns(self, last_value=0.0):
        R = last_value
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns


# ─────────────────────────────────────────────
# PPO Update
# ─────────────────────────────────────────────
def ppo_update(net, optimizer, buffer, last_value=0.0):
    returns   = buffer.compute_returns(last_value)
    states    = torch.tensor(np.array(buffer.states),   dtype=torch.float32)
    actions   = torch.tensor(np.array(buffer.actions),  dtype=torch.float32)
    old_logps = torch.tensor(np.array(buffer.logprobs), dtype=torch.float32)
    values    = torch.tensor(np.array(buffer.values),   dtype=torch.float32)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = states.shape[0]
    total_loss = 0.0

    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(dataset_size)
        for start in range(0, dataset_size, BATCH_SIZE):
            mb = idx[start:start + BATCH_SIZE]
            mb_s   = states[mb]
            mb_a   = actions[mb]
            mb_olp = old_logps[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            new_logp, new_val, entropy = net.evaluate(mb_s, mb_a)
            ratio = torch.exp(new_logp - mb_olp)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss =  0.5 * (new_val - mb_ret).pow(2).mean()
            ent_loss    = -ENTROPY_COEF * entropy.mean()

            loss = actor_loss + critic_loss + ent_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

    buffer.clear()
    return total_loss


# ─────────────────────────────────────────────
# Single Run
# ─────────────────────────────────────────────
def run_simulation(b=1.3, p_vis=1.0, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env    = SpatialPDEnv(N=GRID_N, b=b, p_vis=p_vis)
    net    = ActorCritic().to(DEVICE)
    optim_ = optim.Adam(net.parameters(), lr=LR)
    buffer = RolloutBuffer()

    states = env.reset()
    coop_history = []
    UPDATE_EVERY = 50   # PPO update every N steps

    for t in range(T_STEPS):
        s_tensor = torch.tensor(states, dtype=torch.float32)

        # Decide: PPO or Fermi for each agent
        use_ppo = np.random.rand(N_AGENTS) > EPSILON   # True -> PPO

        with torch.no_grad():
            actions, logps, values = net.get_action_and_logprob(s_tensor)

        ppo_actions = actions.numpy().astype(int)

        next_states, rewards, coop_rate = env.step(ppo_actions, use_ppo, t)
        coop_history.append(coop_rate)

        # Store PPO transitions only for agents that used PPO
        ppo_idx = np.where(use_ppo)[0]
        if len(ppo_idx) > 0:
            buffer.states.append(states[ppo_idx])
            buffer.actions.append(ppo_actions[ppo_idx])
            buffer.logprobs.append(logps.numpy()[ppo_idx])
            buffer.rewards.append(rewards[ppo_idx])
            buffer.values.append(values.numpy()[ppo_idx])

        states = next_states

        # PPO update
        if (t + 1) % UPDATE_EVERY == 0 and len(buffer.states) > 0:
            # Flatten buffer for update
            flat_buf = RolloutBuffer()
            flat_buf.states   = [s for batch in buffer.states   for s in batch]
            flat_buf.actions  = [a for batch in buffer.actions  for a in batch]
            flat_buf.logprobs = [l for batch in buffer.logprobs for l in batch]
            flat_buf.rewards  = [r for batch in buffer.rewards  for r in batch]
            flat_buf.values   = [v for batch in buffer.values   for v in batch]
            ppo_update(net, optim_, flat_buf)
            buffer.clear()

    return np.array(coop_history)


# ─────────────────────────────────────────────
# Experiment 1: Cooperation vs p_vis (fixed b=1.3)
# ─────────────────────────────────────────────
def exp1_visibility_sweep(b=1.3):
    print(f"\n=== Exp1: Visibility Sweep (b={b}) ===")
    results = {}   # p_vis -> mean final coop rate

    for p in P_VIS_RANGE:
        coop_finals = []
        for seed in range(N_SEEDS):
            print(f"  p={p:.1f}, seed={seed}", flush=True)
            hist = run_simulation(b=b, p_vis=p, seed=seed)
            coop_finals.append(hist[-100:].mean())   # average last 100 steps
        results[p] = (np.mean(coop_finals), np.std(coop_finals))
        print(f"  -> mean={results[p][0]:.3f} ± {results[p][1]:.3f}")

    return results


# ─────────────────────────────────────────────
# Experiment 2: Cooperation heatmap (b vs p_vis)
# ─────────────────────────────────────────────
def exp2_heatmap(b_list=None, p_list=None):
    if b_list is None:
        b_list = [1.1, 1.3, 1.5, 1.7]
    if p_list is None:
        p_list = P_VIS_RANGE

    print(f"\n=== Exp2: Heatmap b x p_vis ===")
    matrix = np.zeros((len(b_list), len(p_list)))

    for i, b in enumerate(b_list):
        for j, p in enumerate(p_list):
            runs = []
            for seed in range(N_SEEDS):
                print(f"  b={b}, p={p:.1f}, seed={seed}", flush=True)
                hist = run_simulation(b=b, p_vis=p, seed=seed)
                runs.append(hist[-100:].mean())
            matrix[i, j] = np.mean(runs)

    return matrix, b_list, p_list


# ─────────────────────────────────────────────
# Experiment 3: Temporal dynamics for key p values
# ─────────────────────────────────────────────
def exp3_temporal(b=1.3, p_values=None):
    if p_values is None:
        p_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"\n=== Exp3: Temporal Dynamics (b={b}) ===")
    histories = {}

    for p in p_values:
        runs = []
        for seed in range(N_SEEDS):
            print(f"  p={p:.1f}, seed={seed}", flush=True)
            hist = run_simulation(b=b, p_vis=p, seed=seed)
            runs.append(hist)
        histories[p] = np.array(runs)  # (seeds, T)

    return histories


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_all(exp1_res, heatmap_data, temporal_data, b_fixed=1.3):
    matrix, b_list, p_list = heatmap_data

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Plot 1: Visibility sweep bar chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    ps   = list(exp1_res.keys())
    means = [exp1_res[p][0] for p in ps]
    stds  = [exp1_res[p][1] for p in ps]
    colors = ['#d62728' if m < 0.5 else '#2ca02c' for m in means]
    ax1.bar(ps, means, width=0.07, color=colors, alpha=0.8, yerr=stds, capsize=4)
    ax1.axhline(0.5, color='k', ls='--', lw=1, label='50% threshold')
    ax1.set_xlabel('Visibility Probability p', fontsize=12)
    ax1.set_ylabel('Final Cooperation Rate ρ_C', fontsize=12)
    ax1.set_title(f'Effect of Visibility on Cooperation (b={b_fixed})', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.set_xticks(ps)

    # --- Plot 2: Temporal dynamics ---
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_lines = plt.cm.viridis(np.linspace(0.1, 0.9, len(temporal_data)))
    for (p, hists), col in zip(temporal_data.items(), cmap_lines):
        mean_h = hists.mean(axis=0)
        std_h  = hists.std(axis=0)
        t_arr  = np.arange(len(mean_h))
        ax2.plot(t_arr, mean_h, color=col, label=f'p={p:.1f}', lw=1.5)
        ax2.fill_between(t_arr, mean_h - std_h, mean_h + std_h,
                         color=col, alpha=0.15)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Cooperation Frequency ρ_C', fontsize=12)
    ax2.set_title(f'Temporal Evolution (b={b_fixed})', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(0, 1.05)

    # --- Plot 3: Heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(matrix, aspect='auto', cmap='RdYlGn',
                    vmin=0, vmax=1, origin='lower')
    ax3.set_xticks(range(len(p_list)))
    ax3.set_xticklabels([f'{p:.1f}' for p in p_list], fontsize=9)
    ax3.set_yticks(range(len(b_list)))
    ax3.set_yticklabels([f'{b:.1f}' for b in b_list], fontsize=9)
    ax3.set_xlabel('Visibility Probability p', fontsize=12)
    ax3.set_ylabel('Defection Temptation b', fontsize=12)
    ax3.set_title('Cooperation Rate Heatmap (b × p)', fontsize=12)
    plt.colorbar(im, ax=ax3, label='ρ_C')
    # Annotate cells
    for i in range(len(b_list)):
        for j in range(len(p_list)):
            ax3.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                     fontsize=7, color='black')

    # --- Plot 4: Critical threshold identification ---
    ax4 = fig.add_subplot(gs[1, 1])
    ps_arr   = np.array(list(exp1_res.keys()))
    means_arr = np.array([exp1_res[p][0] for p in ps_arr])
    stds_arr  = np.array([exp1_res[p][1] for p in ps_arr])

    ax4.errorbar(ps_arr, means_arr, yerr=stds_arr, fmt='o-',
                 color='steelblue', capsize=5, lw=2, markersize=6, label='PPO hybrid')
    ax4.fill_between(ps_arr, means_arr - stds_arr, means_arr + stds_arr,
                     alpha=0.2, color='steelblue')
    ax4.axhline(0.5, color='gray', ls='--', lw=1)

    # Find critical point
    crit_p = None
    for k in range(len(means_arr) - 1):
        if means_arr[k] < 0.5 and means_arr[k+1] >= 0.5:
            crit_p = (ps_arr[k] + ps_arr[k+1]) / 2
            break
    if crit_p:
        ax4.axvline(crit_p, color='red', ls=':', lw=2,
                    label=f'p* ≈ {crit_p:.2f} (critical threshold)')

    ax4.set_xlabel('Visibility Probability p', fontsize=12)
    ax4.set_ylabel('Cooperation Rate ρ_C', fontsize=12)
    ax4.set_title(f'Critical Visibility Threshold Detection (b={b_fixed})', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 1.05)

    plt.suptitle('Spatial PD with PPO under Partial Observability\n'
                 '(von Neumann, 100×100, Fermi-PPO Hybrid)',
                 fontsize=13, fontweight='bold')

    out_path = '/mnt/user-data/outputs/spatial_pd_ppo_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {out_path}")
    plt.close()
    return out_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    start = time.time()

    B_FIXED = 1.3

    # Experiment 1: sweep p_vis
    exp1_res = exp1_visibility_sweep(b=B_FIXED)

    # Experiment 2: heatmap (b x p)
    heatmap_data = exp2_heatmap(
        b_list=[1.1, 1.3, 1.5, 1.7],
        p_list=[0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # Experiment 3: temporal dynamics
    temporal_data = exp3_temporal(b=B_FIXED, p_values=[0.2, 0.4, 0.6, 0.8, 1.0])

    # Plot
    fig_path = plot_all(exp1_res, heatmap_data, temporal_data, b_fixed=B_FIXED)

    # Save numerical results
    np.save('/mnt/user-data/outputs/exp1_visibility.npy', exp1_res)
    np.save('/mnt/user-data/outputs/exp2_heatmap.npy',    heatmap_data[0])

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("Done!")