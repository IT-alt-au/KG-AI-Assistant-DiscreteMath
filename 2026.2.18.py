import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 10


class FermiAdaptiveQLearning:
    def __init__(self, L=100, b=1.3, alpha=0.1, gamma=0.95, epsilon=0.1,
                 psi=1.02, M_max=10, q_init_std=0.01,
                 adaptation_start_step=5000, fermi_kappa=0.1):
        self.L = L
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.psi = psi
        self.M_max = M_max
        self.adaptation_start_step = adaptation_start_step
        self.fermi_kappa = fermi_kappa
        self.payoff_matrix = np.array([[1.0, 0.0], [b, 0.0]], dtype=np.float64)
        self.strategies = np.random.randint(0, 2, (L, L), dtype=np.int32)
        self.Q_tables = np.random.randn(L, L, 5, 2) * q_init_std
        self.memory_lengths = np.ones((L, L), dtype=np.int32)
        self.payoff_history = np.zeros((L, L, M_max), dtype=np.float64)
        self.td_target_history = np.zeros((L, L, M_max), dtype=np.float64)
        self.history_count = np.zeros((L, L), dtype=np.int32)
        self.neighbor_indices = self._precompute_neighbors()
        self.weight_cache = self._precompute_weights_array()

    def _precompute_neighbors(self):
        neighbors = np.zeros((self.L, self.L, 4, 2), dtype=np.int32)
        for i in range(self.L):
            for j in range(self.L):
                neighbors[i, j, 0] = [(i - 1) % self.L, j]
                neighbors[i, j, 1] = [(i + 1) % self.L, j]
                neighbors[i, j, 2] = [i, (j - 1) % self.L]
                neighbors[i, j, 3] = [i, (j + 1) % self.L]
        return neighbors

    def _precompute_weights_array(self):
        weights = np.zeros((self.M_max + 1, self.M_max), dtype=np.float64)
        for m in range(1, self.M_max + 1):
            w = np.exp(np.linspace(-1.0, 0.0, m))
            weights[m, :m] = w / np.sum(w)
        return weights

    def step(self, current_global_step):
        states = self._calculate_states(self.strategies, self.neighbor_indices)
        actions = self._decide_actions(
            self.strategies, states, self.Q_tables,
            self.payoff_history, self.history_count, self.neighbor_indices,
            self.epsilon, self.fermi_kappa)
        self._update_core(
            actions, states, self.strategies, self.Q_tables,
            self.payoff_history, self.td_target_history,
            self.memory_lengths, self.history_count,
            self.neighbor_indices, self.payoff_matrix,
            self.weight_cache, self.alpha, self.gamma,
            self.psi, self.M_max,
            current_global_step, self.adaptation_start_step)
        self.strategies = actions.copy()

    @staticmethod
    @jit(nopython=True)
    def _calculate_states(strategies, neighbor_indices):
        L = strategies.shape[0]
        states = np.zeros((L, L), dtype=np.int32)
        for i in range(L):
            for j in range(L):
                state = 0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    if strategies[ni, nj] == 0:
                        state += 1
                states[i, j] = state
        return states

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _decide_actions(strategies, states, Q_tables, payoff_history, history_count,
                        neighbor_indices, epsilon, kappa):
        L = strategies.shape[0]
        actions = np.zeros((L, L), dtype=np.int32)
        for i in prange(L):
            for j in range(L):
                if np.random.rand() < epsilon:
                    k = np.random.randint(0, 4)
                    ni, nj = neighbor_indices[i, j, k]
                    my_ptr = history_count[i, j] - 1
                    neigh_ptr = history_count[ni, nj] - 1
                    if my_ptr >= 0 and neigh_ptr >= 0:
                        pi = payoff_history[i, j, my_ptr]
                        pj = payoff_history[ni, nj, neigh_ptr]
                        fermi_prob = 1.0 / (1.0 + np.exp((pi - pj) / kappa))
                        if np.random.rand() < fermi_prob:
                            actions[i, j] = strategies[ni, nj]
                        else:
                            actions[i, j] = strategies[i, j]
                    else:
                        actions[i, j] = np.random.randint(0, 2)
                else:
                    state = states[i, j]
                    Q_values = Q_tables[i, j, state]
                    if Q_values[0] == Q_values[1]:
                        actions[i, j] = np.random.randint(0, 2)
                    elif Q_values[0] > Q_values[1]:
                        actions[i, j] = 0
                    else:
                        actions[i, j] = 1
        return actions

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_core(actions, states, strategies, Q_tables, payoff_history,
                     td_target_history, memory_lengths, history_count,
                     neighbor_indices, payoff_matrix, weight_cache,
                     alpha, gamma, psi, M_max, current_step, start_step):
        L = strategies.shape[0]
        for i in prange(L):
            for j in range(L):
                state = states[i, j]
                current_action = strategies[i, j]
                payoff = 0.0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    payoff += payoff_matrix[current_action, strategies[ni, nj]]
                count = history_count[i, j]
                if count < M_max:
                    payoff_history[i, j, count] = payoff
                    history_count[i, j] = count + 1
                else:
                    for idx in range(M_max - 1):
                        payoff_history[i, j, idx] = payoff_history[i, j, idx + 1]
                    payoff_history[i, j, M_max - 1] = payoff
                if current_step > start_step:
                    current_memory = memory_lengths[i, j]
                    count_now = history_count[i, j]
                    if count_now >= 4:
                        window = max(2, current_memory // 2)
                        window = min(window, count_now // 2)
                        recent_avg = 0.0
                        for idx in range(count_now - window, count_now):
                            recent_avg += payoff_history[i, j, idx]
                        recent_avg /= window
                        past_avg = 0.0
                        for idx in range(count_now - window):
                            past_avg += payoff_history[i, j, idx]
                        past_avg /= (count_now - window)
                        if past_avg < 0.01: past_avg = 0.01
                        ratio = recent_avg / past_avg
                        if ratio > psi:
                            new_memory = min(current_memory + 1, M_max)
                        elif ratio < (1.0 / psi):
                            new_memory = max(current_memory - 1, 1)
                        else:
                            new_memory = current_memory
                        coop_neighbors = 0
                        for k in range(4):
                            ni, nj = neighbor_indices[i, j, k]
                            if strategies[ni, nj] == 0:
                                coop_neighbors += 1
                        if coop_neighbors >= 3 and current_action == 0:
                            new_memory = min(new_memory + 1, M_max)
                        memory_lengths[i, j] = new_memory
                m = memory_lengths[i, j]
                count_now = history_count[i, j]
                weighted_reward = 0.0
                if count_now > 0:
                    if count_now < m:
                        for idx in range(count_now):
                            weighted_reward += payoff_history[i, j, idx]
                        weighted_reward /= count_now
                    else:
                        for idx in range(m):
                            weighted_reward += payoff_history[i, j, count_now - m + idx] * weight_cache[m, idx]
                next_state_est = 0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    if strategies[ni, nj] == 0:
                        next_state_est += 1
                current_Q = Q_tables[i, j, state, current_action]
                max_next_Q = max(Q_tables[i, j, next_state_est, 0], Q_tables[i, j, next_state_est, 1])
                Q_tables[i, j, state, current_action] += alpha * (weighted_reward + gamma * max_next_Q - current_Q)

    def get_cooperation_frequency(self):
        return np.mean(self.strategies == 0)


def memory_evolution_with_std():
    # === 参数设置（与原实验完全一致）===
    L = 100
    b = 1.3
    psi = 1.1
    epsilon = 0.7
    adaptation_start = 1
    n_seeds = 5
    M_max_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    checkpoints = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                   550, 600, 650, 700, 750, 800, 900, 1000]

    colors = plt.cm.plasma(np.linspace(0, 1, len(M_max_values)))

    print("【Numba Warm-up】Compiling...")
    FermiAdaptiveQLearning(L=10, M_max=5).step(0)
    print("  ✓ Done\n")

    # results[M_max] = {
    #   'steps': [...],
    #   'coop_mean': [...], 'coop_std': [...],
    #   'mem_mean':  [...], 'mem_std':  [...]
    # }
    results = {}

    for idx, M_max in enumerate(M_max_values):
        print(f"M_max={M_max} ({idx+1}/{len(M_max_values)}), running {n_seeds} seeds...")

        all_coop = []   # shape: (n_seeds, len(checkpoints))
        all_mem  = []

        for seed in range(n_seeds):
            np.random.seed(seed)
            model = FermiAdaptiveQLearning(
                L=L, b=b, psi=psi, M_max=M_max,
                epsilon=epsilon,
                adaptation_start_step=adaptation_start
            )

            coop_curve = [model.get_cooperation_frequency()]
            mem_curve  = [np.mean(model.memory_lengths)]
            current_step = 0

            for checkpoint in checkpoints:
                while current_step < checkpoint:
                    current_step += 1
                    model.step(current_step)
                coop_curve.append(model.get_cooperation_frequency())
                mem_curve.append(np.mean(model.memory_lengths))

            all_coop.append(coop_curve)
            all_mem.append(mem_curve)

        all_coop = np.array(all_coop)  # (n_seeds, len(checkpoints)+1)
        all_mem  = np.array(all_mem)

        steps_list = [0] + list(checkpoints)

        results[M_max] = {
            'steps':     steps_list,
            'coop_mean': all_coop.mean(axis=0),
            'coop_std':  all_coop.std(axis=0),
            'mem_mean':  all_mem.mean(axis=0),
            'mem_std':   all_mem.std(axis=0),
            'color':     colors[idx]
        }
        print(f"  final coop: {all_coop[:,-1].mean():.3f} ± {all_coop[:,-1].std():.3f}")

    # === 画图（与原版布局完全一致，仅增加误差带）===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.7, 9), dpi=150)

    fig.suptitle(
        f'Fermi-Q Hybrid Learning with Adaptive Memory\n'
        f'b={b}, ψ={psi}, ε={epsilon}, Grid={L}×{L}',
        fontsize=12, fontweight='bold', y=0.98
    )

    # 子图(a)：合作频率 + 误差带
    for M in M_max_values:
        d = results[M]
        ax1.plot(d['steps'], d['coop_mean'],
                 label=f'M={M}', color=d['color'],
                 linewidth=1.8, marker='o', markersize=2.5)
        ax1.fill_between(d['steps'],
                         d['coop_mean'] - d['coop_std'],
                         d['coop_mean'] + d['coop_std'],
                         alpha=0.12, color=d['color'])

    ax1.set_ylabel('Cooperation Frequency', fontweight='bold')
    ax1.set_xlabel('Time Steps', fontweight='bold')
    ax1.set_title('(a) Evolution of Cooperation', loc='center', y=-0.32,
                  fontweight='bold', fontsize=11)
    ax1.legend(ncol=3, loc='upper left', fontsize=6.5, labelspacing=0.25)
    ax1.axvline(adaptation_start, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.25)

    # 子图(b)：记忆长度 + 误差带
    for M in M_max_values:
        d = results[M]
        ax2.plot(d['steps'], d['mem_mean'],
                 label=f'M={M}', color=d['color'],
                 linewidth=1.8, marker='o', markersize=2.5)
        ax2.fill_between(d['steps'],
                         d['mem_mean'] - d['mem_std'],
                         d['mem_mean'] + d['mem_std'],
                         alpha=0.12, color=d['color'])

    ax2.set_ylabel('Average Memory Length', fontweight='bold')
    ax2.set_xlabel('Time Steps', fontweight='bold')
    ax2.set_title('(b) Evolution of Memory Length', loc='center', y=-0.32,
                  fontweight='bold', fontsize=11)
    ax2.axvline(adaptation_start, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.25)

    plt.subplots_adjust(
        hspace=0.55, bottom=0.18, top=0.92, left=0.12, right=0.95
    )

    filename = f"fermi_q_memory_evolution_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"\n✓ Saved: {filename}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    memory_evolution_with_std()