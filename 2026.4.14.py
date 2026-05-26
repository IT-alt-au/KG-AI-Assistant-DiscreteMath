import numpy as np
from numba import jit, prange


class FermiAdaptiveQLearning:
    def __init__(self, L=100, b=1.3, alpha=0.1, gamma=0.95, epsilon=0.7,
                 psi=1.1, M_max=10, q_init_std=0.01,
                 adaptation_start_step=1, fermi_kappa=0.1):
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
            self.epsilon, self.fermi_kappa
        )
        self._update_core(
            actions, states, self.strategies, self.Q_tables,
            self.payoff_history, self.td_target_history,
            self.memory_lengths, self.history_count,
            self.neighbor_indices, self.payoff_matrix,
            self.weight_cache, self.alpha, self.gamma,
            self.psi, self.M_max,
            current_global_step, self.adaptation_start_step
        )
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
                        window = max(3, current_memory // 2)
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


# ── 消融变体1：固定M=4，去掉自适应，保留Fermi+Q ──────────────────────────
class FermiQ_FixedMemory(FermiAdaptiveQLearning):
    def __init__(self, L=100, b=1.3, fixed_M=4, **kwargs):
        super().__init__(L=L, b=b, M_max=fixed_M,
                         adaptation_start_step=999_999_999, **kwargs)
        self.fixed_M = fixed_M
        self.memory_lengths = np.full((L, L), fixed_M, dtype=np.int32)

    def step(self, t):
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
            self.psi, self.M_max, t, self.adaptation_start_step)
        self.memory_lengths[:] = self.fixed_M   # 强制固定
        self.strategies = actions.copy()


# ── 运行消融实验 ───────────────────────────────────────────────────────────
def run():
    L, T, n_seeds = 100, 1000, 5
    psi, epsilon = 1.1, 0.7
    b_values = [1.3, 1.5]

    # 四种配置：名称 → 构造函数
    configs = {
        'Baseline (Static M=1, Fermi+Q)':      lambda b: FermiAdaptiveQLearning(
            L=L, b=b, M_max=1, psi=psi, epsilon=epsilon,
            adaptation_start_step=999_999_999),
        'Static M=4, Fermi+Q (No Adaptive)':   lambda b: FermiQ_FixedMemory(
            L=L, b=b, fixed_M=4, psi=psi, epsilon=epsilon),
        'Proposed Hybrid M_max=4 (Full)':      lambda b: FermiAdaptiveQLearning(
            L=L, b=b, M_max=4, psi=psi, epsilon=epsilon,
            adaptation_start_step=1),
        'Proposed Hybrid M_max=10 (Full)':     lambda b: FermiAdaptiveQLearning(
            L=L, b=b, M_max=10, psi=psi, epsilon=epsilon,
            adaptation_start_step=1),
    }

    # Numba预热
    print("Numba warming up...", flush=True)
    FermiAdaptiveQLearning(L=10, M_max=4).step(0)
    FermiQ_FixedMemory(L=10, b=1.3, fixed_M=4).step(0)
    print("Done.\n")

    table = {}  # {config_name: {b: (mean, std)}}

    for name, fn in configs.items():
        table[name] = {}
        for b in b_values:
            finals = []
            for seed in range(n_seeds):
                np.random.seed(seed)
                model = fn(b)
                for t in range(1, T + 1):
                    model.step(t)
                finals.append(model.get_cooperation_frequency())
            mean, std = np.mean(finals), np.std(finals)
            table[name][b] = (mean, std)
            print(f"  {name} | b={b} → {mean:.1%} ± {std:.1%}")
        print()

    # ── 打印最终表格 ──────────────────────────────────────────────────────
    print("=" * 68)
    print(f"{'Model / Mechanism':<42} {'ρC(b=1.3)':>12} {'ρC(b=1.5)':>12}")
    print("-" * 68)
    for name, res in table.items():
        m13, s13 = res[1.3]
        m15, s15 = res[1.5]
        print(f"{name:<42} {m13:>10.1%}±{s13:.1%}  {m15:>10.1%}±{s15:.1%}")
    print("=" * 68)


if __name__ == "__main__":
    run()