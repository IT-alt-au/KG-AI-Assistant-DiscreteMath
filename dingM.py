import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


class FermiFixedMemory:
    """
    【基准模型：固定记忆】
    用于生成对比数据。记忆长度 M 是固定的，不演化。
    """

    def __init__(self, L=100, b=1.3, alpha=0.1, gamma=0.95, epsilon=0.1,
                 M_fixed=1,
                 q_init_std=0.01,
                 fermi_kappa=0.1):

        self.L = L
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.M_fixed = int(M_fixed)
        self.fermi_kappa = fermi_kappa

        self.payoff_matrix = np.array([[1.0, 0.0], [b, 0.0]], dtype=np.float64)
        self.strategies = np.random.randint(0, 2, (L, L), dtype=np.int32)
        self.Q_tables = np.random.randn(L, L, 5, 2) * q_init_std

        # 【差异点】记忆长度固定为 M_fixed，永远不变
        self.memory_lengths = np.ones((L, L), dtype=np.int32) * self.M_fixed

        self.payoff_history = np.zeros((L, L, self.M_fixed), dtype=np.float64)
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
        weights = np.zeros((self.M_fixed + 1, self.M_fixed), dtype=np.float64)
        for m in range(1, self.M_fixed + 1):
            w = np.exp(np.linspace(-1.0, 0.0, m))
            w_normalized = w / np.sum(w)
            weights[m, :m] = w_normalized
        return weights

    def step(self):
        states = self._calculate_states(self.strategies, self.neighbor_indices)
        actions = self._decide_actions(
            self.strategies, states, self.Q_tables,
            self.payoff_history, self.history_count, self.neighbor_indices,
            self.epsilon, self.fermi_kappa
        )
        self._update_core(
            actions, states, self.strategies, self.Q_tables,
            self.payoff_history, self.memory_lengths, self.history_count,
            self.neighbor_indices, self.payoff_matrix,
            self.weight_cache, self.alpha, self.gamma,
            self.M_fixed
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
                if np.random.rand() < epsilon:  # Fermi
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
                else:  # Q-Learning
                    state = states[i, j]
                    Q_values = Q_tables[i, j, state]
                    if Q_values[0] == Q_values[1]:
                        actions[i, j] = np.random.randint(0, 2)
                    else:
                        if Q_values[0] > Q_values[1]:
                            actions[i, j] = 0
                        else:
                            actions[i, j] = 1
        return actions

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_core(actions, states, strategies, Q_tables, payoff_history,
                     memory_lengths, history_count,
                     neighbor_indices, payoff_matrix, weight_cache,
                     alpha, gamma, M_fixed):
        L = strategies.shape[0]
        for i in prange(L):
            for j in range(L):
                state = states[i, j]
                current_action = strategies[i, j]
                payoff = 0.0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    payoff += payoff_matrix[current_action, strategies[ni, nj]]

                # 固定记忆历史更新
                count = history_count[i, j]
                if count < M_fixed:
                    payoff_history[i, j, count] = payoff
                    history_count[i, j] = count + 1
                else:
                    for idx in range(M_fixed - 1):
                        payoff_history[i, j, idx] = payoff_history[i, j, idx + 1]
                    payoff_history[i, j, M_fixed - 1] = payoff

                # 这里没有自适应调整，m 始终为 M_fixed
                m = M_fixed

                # 计算加权奖励
                count_now = history_count[i, j]
                weighted_reward = 0.0
                if count_now > 0:
                    if count_now < m:
                        for idx in range(count_now): weighted_reward += payoff_history[i, j, idx]
                        weighted_reward /= count_now
                    else:
                        for idx in range(m):
                            weighted_reward += payoff_history[i, j, count_now - m + idx] * weight_cache[m, idx]

                # Q-Learning 更新
                next_state_est = 0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    if strategies[ni, nj] == 0: next_state_est += 1

                current_Q = Q_tables[i, j, state, current_action]
                max_next_Q = max(Q_tables[i, j, next_state_est, 0], Q_tables[i, j, next_state_est, 1])
                td_target = weighted_reward + gamma * max_next_Q
                td_error = td_target - current_Q
                Q_tables[i, j, state, current_action] += alpha * td_error

    def get_cooperation_frequency(self):
        return np.mean(self.strategies == 0)


def plot_fixed_memory_benchmark():
    # === 核心参数设置 ===
    L = 100
    epsilon = 0.7  # 保持和您之前实验一致，控制变量

    # 扫描的 b 值 (每条线)
    b_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # 扫描的 M 值 (横坐标)
    M_values = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    steps_per_run = 15000  # 运行步数
    avg_window = 15000  # 统计最后多少步的平均值

    print(f"\n{'=' * 80}")
    print(f"Generating Benchmark Plot: Fixed Memory vs Cooperation")
    print(f"Fixed epsilon: {epsilon}")
    print(f"{'=' * 80}")

    # Numba Warm-up
    FermiFixedMemory(L=10, M_fixed=5).step()

    # 存储数据
    results = {b: [] for b in b_values}

    for b_val in b_values:
        print(f"\n>>> Simulating Curve for b = {b_val}")
        for M_val in M_values:
            print(f"    Running Fixed M={M_val} ... ", end="")

            model = FermiFixedMemory(L=L, b=b_val, epsilon=epsilon, M_fixed=M_val)

            # 跑模拟
            recent_coops = []
            for s in range(steps_per_run):
                model.step()
                if s > (steps_per_run - avg_window):
                    recent_coops.append(model.get_cooperation_frequency())

            final_coop = np.mean(recent_coops)
            results[b_val].append(final_coop)
            print(f"Coop = {final_coop:.4f}")

    # === 绘图 ===
    print("\n【Visualization】Creating plots...")
    plt.figure(figsize=(10, 7), dpi=150)

    markers = ['o', 's', '^', 'D', 'v', 'x']

    for i, b_val in enumerate(b_values):
        plt.plot(M_values, results[b_val],
                 label=f'b={b_val}',
                 marker=markers[i % len(markers)],
                 linewidth=2, alpha=0.8)

    plt.xlabel('Fixed Memory Length ($M$)', fontweight='bold', fontsize=12)
    plt.ylabel('Final Cooperation Frequency', fontweight='bold', fontsize=12)
    plt.title(f'Effect of Fixed Memory Length on Cooperation\n(Benchmark: $\epsilon$={epsilon})', y=-0.2,
              fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.legend(title="Temptation (b)", loc='best')
    plt.xticks(M_values)  # 显示所有M刻度
    plt.ylim(-0.05, 1.05)

    plt.subplots_adjust(bottom=0.15)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_fixed_M_vs_Coop_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


if __name__ == "__main__":
    plot_fixed_memory_benchmark()