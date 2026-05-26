import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 引入颜色模块
import matplotlib
from datetime import datetime
import os

# 设置matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


class FermiAdaptiveQLearning:
    """
    Fermi-Q Learning with Adaptive Memory
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
        self.epsilon = epsilon
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
        L = self.L
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
                    else:
                        if Q_values[0] > Q_values[1]:
                            actions[i, j] = 0
                        else:
                            actions[i, j] = 1
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
                current_action = strategies[i, j]  # Payoff based on current strategy

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
                        for idx in range(count_now): weighted_reward += payoff_history[i, j, idx]
                        weighted_reward /= count_now
                    else:
                        for idx in range(m):
                            weighted_reward += payoff_history[i, j, count_now - m + idx] * weight_cache[m, idx]

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


# --- 新增：快照绘制函数 ---
def save_snapshot(strategies, step, M_max, run_id):
    """
    保存当前的博弈网格快照
    0 (Cooperator) -> Blue
    1 (Defector)   -> Red
    """
    plt.figure(figsize=(6, 6), dpi=100)

    # 自定义颜色映射：蓝色合作，红色背叛
    cmap = mcolors.ListedColormap(['#1f77b4', '#d62728'])  # Standard Matplotlib Blue/Red
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(strategies, cmap=cmap, norm=norm, interpolation='nearest')

    # 移除坐标轴
    plt.axis('off')

    # 标题放在底部
    plt.title(f"Step: {step}, M={M_max} (Blue: Coop, Red: Defect)", y=-0.1, fontsize=12)

    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")

    filename = f"snapshots/snapshot_M{M_max}_step{step}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # print(f"    [Snapshot] Saved {filename}")


def memory_length_scan():
    L = 100
    b = 1.01
    psi = 1.0001
    epsilon = 0.02

    M_max_values = [1, 3, 5, 7, 9, 10, 11, 13, 14, 15]
    total_steps = 100000
    adaptation_start = 5000

    # 我们只想对其中一个最具代表性的参数组生成快照，避免图片过多
    # 这里选择 M_max 最大的那一组，通常它的演化过程最完整
    snapshot_target_M = M_max_values[-1]
    # 定义快照的生成时刻：初始、预热结束、突变后、中期、结束
    snapshot_steps = [0, 100, 5000, 5500, 10000, 50000, 200000]

    checkpoints = [100, 1000, 3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 45000, 50000, 55000, 60000, 65000,
                   70000, 75000, 80000, 85000, 90000, 95000, 100000, 110000, 120000, 125000, 130000, 135000, 140000,
                   145000, 150000, 155000, 160000, 165000, 170000, 175000, 180000, 185000, 190000, 195000, 200000]

    # 确保快照点在检查点列表中，方便循环逻辑（或者在循环内单独判断）
    # 简单起见，我们在循环内部用 current_step 判断是否在 snapshot_steps 里

    results = {}
    colors = plt.cm.plasma(np.linspace(0, 1, len(M_max_values)))

    print("\n【Numba Warm-up】Compiling...")
    FermiAdaptiveQLearning(L=10, M_max=5).step(0)
    print("  ✓ Compilation complete\n")

    for idx, M_max in enumerate(M_max_values):
        print(f"{'─' * 80}")
        print(f"Running: M_max = {M_max} ({idx + 1}/{len(M_max_values)})")
        print(f"{'─' * 80}")

        model = FermiAdaptiveQLearning(
            L=L, b=b, psi=psi, M_max=M_max,
            epsilon=epsilon,
            adaptation_start_step=adaptation_start
        )

        steps_list = [0]
        coop_list = [model.get_cooperation_frequency()]
        mem_avg_list = [np.mean(model.memory_lengths)]

        # 初始快照
        if M_max == snapshot_target_M and 0 in snapshot_steps:
            save_snapshot(model.strategies, 0, M_max, idx)

        current_step = 0

        # 优化打印
        # print(f"{'Step':<10} {'Cooperation':<15} ...")

        for checkpoint in checkpoints:
            # 运行到下一个 checkpoint
            while current_step < checkpoint:
                current_step += 1
                model.step(current_step)

                # --- 快照生成逻辑 ---
                if M_max == snapshot_target_M and current_step in snapshot_steps:
                    save_snapshot(model.strategies, current_step, M_max, idx)
                    print(f"    >> Snapshot saved at step {current_step}")

            c = model.get_cooperation_frequency()
            m = np.mean(model.memory_lengths)
            q = model.get_q_statistics()['q_diff']

            steps_list.append(current_step)
            coop_list.append(c)
            mem_avg_list.append(m)

            # 简略打印以节省屏幕空间
            if current_step % 5000 == 0 or current_step < 10000:
                print(f"Step {current_step}: Coop={c:.3f}, Mem={m:.2f}")

        results[M_max] = {
            'steps': steps_list, 'cooperation': coop_list,
            'mem_avg': mem_avg_list, 'color': colors[idx]
        }
        print()

    # Visualization
    print("\n【Visualization】Creating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150)  # 稍微增加高度以容纳底部标题

    for M in M_max_values:
        d = results[M]
        ax1.plot(d['steps'], d['cooperation'], label=f'M={M}', color=d['color'], linewidth=2)
        ax2.plot(d['steps'], d['mem_avg'], color=d['color'], linewidth=2)

    # --- 修改 1：标题下移 (y=-0.15) ---
    ax1.set_ylabel('Cooperation Frequency', fontweight='bold')
    ax1.set_title('(a) Evolution of Cooperation (Fermi-Q Hybrid)', y=-0.15, fontweight='bold', fontsize=12)
    ax1.legend(ncol=2, loc='upper left')
    ax1.axvline(adaptation_start, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Average Memory Length', fontweight='bold')
    ax2.set_xlabel('Time Steps', fontweight='bold')  # X轴标签保留
    ax2.set_title('(b) Evolution of Memory Length', y=-0.25, fontweight='bold', fontsize=12)  # 标题放更下面
    ax2.axvline(adaptation_start, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 调整布局，增加 hspace 给中间的标题留空间，增加 bottom 给底部的标题留空间
    plt.subplots_adjust(hspace=0.4, bottom=0.15)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"fermi_q_hybrid_{timestamp}.png", bbox_inches='tight')
    print(f"  ✓ Saved: fermi_q_hybrid_{timestamp}.png")
    plt.show()


if __name__ == "__main__":
    memory_length_scan()