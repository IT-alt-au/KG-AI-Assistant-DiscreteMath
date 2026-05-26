import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置matplotlib字体和大小
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


class FermiAdaptiveQLearning:
    """
    Fermi-Q Learning with Adaptive Memory
    结合了演化博弈的模仿动力学(Fermi)与强化学习的自适应记忆(Q-Learning)
    """

    def __init__(self, L=100, b=1.3, alpha=0.1, gamma=0.95, epsilon=0.1,
                 psi=1.02,
                 M_max=10, q_init_std=0.01,
                 adaptation_start_step=5000,
                 fermi_kappa=0.1):

        self.L = L
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # 社会模仿概率
        self.psi = psi
        self.M_max = M_max
        self.adaptation_start_step = adaptation_start_step
        self.fermi_kappa = fermi_kappa

        self.payoff_matrix = np.array([[1.0, 0.0], [b, 0.0]], dtype=np.float64)
        self.strategies = np.random.randint(0, 2, (L, L), dtype=np.int32)
        self.Q_tables = np.random.randn(L, L, 5, 2) * q_init_std

        # 强制从短视开始 (Memory=1)
        self.memory_lengths = np.ones((L, L), dtype=np.int32)

        # 动态分配历史数组
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
            w_normalized = w / np.sum(w)
            weights[m, :m] = w_normalized
        return weights

    def step(self, current_global_step):
        # 1. 计算状态
        states = self._calculate_states(self.strategies, self.neighbor_indices)

        # 2. 决策阶段 (混合策略)
        actions = self._decide_actions(
            self.strategies, states, self.Q_tables,
            self.payoff_history, self.history_count, self.neighbor_indices,
            self.epsilon, self.fermi_kappa
        )

        # 3. 核心更新阶段 (更新Q表和记忆)
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
                # 随机数 < epsilon -> 进行 Fermi 模仿
                if np.random.rand() < epsilon:
                    k = np.random.randint(0, 4)
                    ni, nj = neighbor_indices[i, j, k]

                    my_ptr = history_count[i, j] - 1
                    neigh_ptr = history_count[ni, nj] - 1

                    if my_ptr >= 0 and neigh_ptr >= 0:
                        pi = payoff_history[i, j, my_ptr]
                        pj = payoff_history[ni, nj, neigh_ptr]

                        # Fermi 更新公式
                        fermi_prob = 1.0 / (1.0 + np.exp((pi - pj) / kappa))

                        if np.random.rand() < fermi_prob:
                            actions[i, j] = strategies[ni, nj]
                        else:
                            actions[i, j] = strategies[i, j]
                    else:
                        actions[i, j] = np.random.randint(0, 2)
                else:
                    # 否则 -> 使用 Q-Learning 贪婪策略
                    state = states[i, j]
                    Q_values = Q_tables[i, j, state]

                    if Q_values[0] == Q_values[1]:
                        actions[i, j] = np.random.randint(0, 2)
                    else:
                        if Q_values[0] > Q_values[1]:
                            actions[i, j] = 0  # Coop
                        else:
                            actions[i, j] = 1  # Defect
        return actions

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_core(actions, states, strategies, Q_tables, payoff_history,
                     td_target_history, memory_lengths, history_count,
                     neighbor_indices, payoff_matrix, weight_cache,
                     alpha, gamma, psi, M_max,
                     current_step, start_step):
        L = strategies.shape[0]

        for i in prange(L):
            for j in range(L):
                state = states[i, j]
                current_action = strategies[i, j]

                # --- 1. 计算收益 ---
                payoff = 0.0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    payoff += payoff_matrix[current_action, strategies[ni, nj]]

                # --- 2. 更新历史 ---
                count = history_count[i, j]
                if count < M_max:
                    payoff_history[i, j, count] = payoff
                    history_count[i, j] = count + 1
                else:
                    for idx in range(M_max - 1):
                        payoff_history[i, j, idx] = payoff_history[i, j, idx + 1]
                    payoff_history[i, j, M_max - 1] = payoff

                # --- 3. 自适应记忆调整 ---
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

                        # 辅助机制：如果周围合作环境好，鼓励长记忆
                        coop_neighbors = 0
                        for k in range(4):
                            ni, nj = neighbor_indices[i, j, k]
                            if strategies[ni, nj] == 0:
                                coop_neighbors += 1

                        if coop_neighbors >= 3 and current_action == 0:
                            new_memory = min(new_memory + 1, M_max)

                        memory_lengths[i, j] = new_memory

                # --- 4. 计算 Reward (加权) ---
                m = memory_lengths[i, j]
                count_now = history_count[i, j]
                weighted_reward = 0.0
                if count_now > 0:
                    if count_now < m:
                        for idx in range(count_now): weighted_reward += payoff_history[i, j, idx]
                        weighted_reward /= count_now
                    else:
                        for idx in range(m):
                            weighted_reward += payoff_history[i, j, count_now - m + idx] * weight_cache[m, idx]

                # --- 5. Q-Learning 更新 ---
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

    def get_q_statistics(self):
        q_coop = self.Q_tables[:, :, :, 0]
        q_defect = self.Q_tables[:, :, :, 1]
        q_diff = np.mean(q_coop - q_defect)
        return {'q_diff': q_diff}


def epsilon_scan_experiment():
    """
    实验1: 固定M_max，扫描epsilon（社会模仿概率）
    目标：找到最优的社会学习-理性决策混合比例
    """
    # === 固定参数 ===
    L = 100
    b = 1.3
    psi = 1.1
    M_max = 10  # 固定足够大的记忆容量

    # === 扫描参数 ===
    epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # === 实验设置 ===
    total_steps = 10000
    adaptation_start = 1
    # 密集检查点用于观察演化过程
    checkpoints = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000,2000, 3000, 5000, 7000, 10000]

    print(f"\n{'=' * 80}")
    print("Experiment 1: Epsilon Scan (Fixed Memory Capacity)")
    print(f"{'=' * 80}")
    print(f"  Grid size L:              {L}×{L}")
    print(f"  Temptation b:             {b}")
    print(f"  Adaptive threshold ψ:     {psi}")
    print(f"  Fixed M_max:              {M_max}")
    print(f"  Total steps:              {total_steps}")
    print(f"  Epsilon range:            {epsilon_values[0]} - {epsilon_values[-1]}")
    print("=" * 80)

    # === Numba预热 ===
    print("\n【Numba Warm-up】Compiling...")
    FermiAdaptiveQLearning(L=10, M_max=5, epsilon=0.5).step(0)
    print("  ✓ Compilation complete\n")

    # === 存储结果 ===
    results = {}
    final_cooperation = []
    final_memory = []

    # === 主实验循环 ===
    for idx, epsilon in enumerate(epsilon_values):
        print(f"{'─' * 80}")
        print(f"Running: ε = {epsilon:.1f} ({idx + 1}/{len(epsilon_values)})")
        print(f"{'─' * 80}")

        model = FermiAdaptiveQLearning(
            L=L, b=b, psi=psi, M_max=M_max,
            epsilon=epsilon,
            adaptation_start_step=adaptation_start
        )

        steps_list = [0]
        coop_list = [model.get_cooperation_frequency()]
        mem_avg_list = [np.mean(model.memory_lengths)]

        current_step = 0

        print(f"{'Step':<10} {'Cooperation':<15} {'Avg Memory':<15}")
        print(f"{0:<10} {coop_list[0]:<15.4f} {mem_avg_list[0]:<15.2f}")

        # 运行模拟
        for checkpoint in checkpoints:
            while current_step < checkpoint:
                current_step += 1
                model.step(current_step)

            c = model.get_cooperation_frequency()
            m = np.mean(model.memory_lengths)

            steps_list.append(current_step)
            coop_list.append(c)
            mem_avg_list.append(m)

            print(f"{current_step:<10} {c:<15.4f} {m:<15.2f}")

        # 存储结果
        results[epsilon] = {
            'steps': steps_list,
            'cooperation': coop_list,
            'mem_avg': mem_avg_list
        }

        final_cooperation.append(coop_list[-1])
        final_memory.append(mem_avg_list[-1])
        print()

    # === 可视化 ===
    print("\n【Visualization】Creating plots...")

    fig = plt.figure(figsize=(14, 10), dpi=150)

    # 整体标题
    fig.suptitle(
        f'Experiment 1: Social Imitation Probability (ε) Scan\n'
        f'Fixed: M_max={M_max}, b={b}, ψ={psi}, Grid={L}×{L}',
        fontsize=14,
        fontweight='bold',
        y=0.97
    )

    # === 子图1: 合作率随时间演化 ===
    ax1 = plt.subplot(2, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))

    for idx, epsilon in enumerate(epsilon_values):
        d = results[epsilon]
        ax1.plot(d['steps'], d['cooperation'],
                 label=f'ε={epsilon:.1f}',
                 color=colors[idx],
                 linewidth=2,
                 marker='o',
                 markersize=4)

    ax1.set_xlabel('Time Steps', fontweight='bold')
    ax1.set_ylabel('Cooperation Frequency', fontweight='bold')
    ax1.set_title('(a) Evolution of Cooperation for Different ε',
                  loc='left', fontweight='bold')
    ax1.legend(ncol=2, fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0, 1.05)

    # === 子图2: 最终合作率 vs ε ===
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epsilon_values, final_cooperation,
             'o-', linewidth=3, markersize=10,
             color='crimson', markerfacecolor='orange',
             markeredgewidth=2, markeredgecolor='crimson')

    # 标注最优点
    max_idx = np.argmax(final_cooperation)
    ax2.annotate(f'Max: ε={epsilon_values[max_idx]:.1f}\nCoop={final_cooperation[max_idx]:.3f}',
                 xy=(epsilon_values[max_idx], final_cooperation[max_idx]),
                 xytext=(epsilon_values[max_idx] - 0.2, final_cooperation[max_idx] - 0.15),
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 color='red', lw=2))

    ax2.set_xlabel('Social Imitation Probability (ε)', fontweight='bold')
    ax2.set_ylabel('Final Cooperation Frequency', fontweight='bold')
    ax2.set_title('(b) Final Cooperation vs ε', loc='left', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 1.05)

    # 添加参考线
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend(fontsize=9)

    # === 子图3: 记忆长度随时间演化 ===
    ax3 = plt.subplot(2, 2, 3)

    for idx, epsilon in enumerate(epsilon_values):
        d = results[epsilon]
        ax3.plot(d['steps'], d['mem_avg'],
                 color=colors[idx],
                 linewidth=2,
                 marker='s',
                 markersize=4,
                 alpha=0.7)

    ax3.set_xlabel('Time Steps', fontweight='bold')
    ax3.set_ylabel('Average Memory Length', fontweight='bold')
    ax3.set_title('(c) Evolution of Memory Length', loc='left', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, total_steps)

    # === 子图4: 最终记忆长度 vs ε ===
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(epsilon_values, final_memory,
             's-', linewidth=3, markersize=10,
             color='steelblue', markerfacecolor='lightblue',
             markeredgewidth=2, markeredgecolor='steelblue')

    ax4.set_xlabel('Social Imitation Probability (ε)', fontweight='bold')
    ax4.set_ylabel('Final Average Memory Length', fontweight='bold')
    ax4.set_title('(d) Final Memory Length vs ε', loc='left', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.axhline(y=M_max, color='red', linestyle='--', alpha=0.5,
                label=f'M_max={M_max}')
    ax4.legend(fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.25)

    # === 保存图片 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"epsilon_scan_M{M_max}_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")

    # === 打印结果摘要 ===
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'ε':<8} {'Final Coop':<15} {'Final Memory':<15} {'Category':<20}")
    print("-" * 80)

    for idx, eps in enumerate(epsilon_values):
        coop = final_cooperation[idx]
        mem = final_memory[idx]

        if coop > 0.9:
            category = "Full Cooperation"
        elif coop > 0.6:
            category = "High Cooperation"
        elif coop > 0.4:
            category = "Medium Cooperation"
        else:
            category = "Low Cooperation"

        marker = "★" if idx == max_idx else " "
        print(f"{eps:<8.1f} {coop:<15.4f} {mem:<15.2f} {category:<20} {marker}")

    print("=" * 80)
    print(f"Optimal ε: {epsilon_values[max_idx]:.1f} (Cooperation: {final_cooperation[max_idx]:.4f})")
    print("=" * 80)

    plt.show()

    return results, epsilon_values, final_cooperation, final_memory


if __name__ == "__main__":
    results, epsilon_vals, final_coop, final_mem = epsilon_scan_experiment()
