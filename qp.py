import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


class DeepFusionQLearning:
    """深度融合Q学习（使用老师的累积收益不等式记忆方法）"""

    def __init__(self, L=100, b=1.3, alpha=0.1, gamma=0.9, epsilon=0.02,
                 alpha_r=0.5, psi=1.1, M_max=10, q_init_std=0.01):

        self.L = L
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_r = alpha_r
        self.psi = psi
        self.M_max = M_max

        self.payoff_matrix = np.array([[1.0, 0.0], [b, 0.0]], dtype=np.float64)
        self.strategies = np.random.randint(0, 2, (L, L), dtype=np.int32)
        self.Q_tables = np.random.randn(L, L, 5, 2) * q_init_std
        self.memory_lengths = np.full((L, L), M_max, dtype=np.int32)

        self.payoff_history = np.zeros((L, L, M_max), dtype=np.float64)
        self.td_target_history = np.zeros((L, L, M_max), dtype=np.float64)
        self.history_count = np.zeros((L, L), dtype=np.int32)

        self.neighbor_indices = self._precompute_neighbors()
        self.weight_cache = self._precompute_weights_array()

    def _precompute_neighbors(self):
        """预计算所有位置的邻居索引"""
        neighbors = np.zeros((self.L, self.L, 4, 2), dtype=np.int32)
        for i in range(self.L):
            for j in range(self.L):
                neighbors[i, j, 0] = [(i - 1) % self.L, j]
                neighbors[i, j, 1] = [(i + 1) % self.L, j]
                neighbors[i, j, 2] = [i, (j - 1) % self.L]
                neighbors[i, j, 3] = [i, (j + 1) % self.L]
        return neighbors

    def _precompute_weights_array(self):
        """预计算权重"""
        weights = np.zeros((self.M_max + 1, self.M_max), dtype=np.float64)
        for m in range(1, self.M_max + 1):
            w = np.exp(np.linspace(-1.0, 0.0, m))
            w_normalized = w / np.sum(w)
            weights[m, :m] = w_normalized
        return weights

    def step(self):
        """单步演化"""
        L = self.L

        actions = np.zeros((L, L), dtype=np.int32)
        states = np.zeros((L, L), dtype=np.int32)

        for i in range(L):
            for j in range(L):
                state = 0
                for k in range(4):
                    ni, nj = self.neighbor_indices[i, j, k]
                    if self.strategies[ni, nj] == 0:
                        state += 1
                states[i, j] = state

                if np.random.rand() < self.epsilon:
                    actions[i, j] = np.random.randint(0, 2)
                else:
                    Q_values = self.Q_tables[i, j, state]
                    max_Q = np.max(Q_values)
                    max_actions = np.where(Q_values == max_Q)[0]
                    actions[i, j] = np.random.choice(max_actions)

        self._update_core_loop(
            actions, states, self.strategies, self.Q_tables,
            self.payoff_history, self.td_target_history,
            self.memory_lengths, self.history_count,
            self.neighbor_indices, self.payoff_matrix,
            self.weight_cache, self.alpha, self.gamma,
            self.alpha_r, self.psi, self.M_max
        )

        self.strategies = actions.copy()

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_core_loop(actions, states, strategies, Q_tables, payoff_history,
                          td_target_history, memory_lengths, history_count,
                          neighbor_indices, payoff_matrix, weight_cache,
                          alpha, gamma, alpha_r, psi, M_max):
        """Numba加速的核心更新循环"""
        L = strategies.shape[0]

        for i in prange(L):
            for j in range(L):
                state = states[i, j]
                action = actions[i, j]

                strategy = strategies[i, j]
                payoff = 0.0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    payoff += payoff_matrix[strategy, strategies[ni, nj]]

                count = history_count[i, j]
                if count < M_max:
                    payoff_history[i, j, count] = payoff
                    history_count[i, j] = count + 1
                else:
                    for idx in range(M_max - 1):
                        payoff_history[i, j, idx] = payoff_history[i, j, idx + 1]
                    payoff_history[i, j, M_max - 1] = payoff

                count_now = history_count[i, j]

                if count_now >= M_max:
                    new_memory = M_max

                    for m_test in range(1, M_max + 1):
                        recent_sum = 0.0
                        for idx in range(count_now - m_test, count_now):
                            recent_sum += payoff_history[i, j, idx]

                        past_sum = 0.0
                        for idx in range(count_now - M_max, count_now - m_test):
                            past_sum += payoff_history[i, j, idx]

                        if past_sum < 0.01:
                            past_sum = 0.01

                        if recent_sum >= psi * past_sum:
                            new_memory = m_test
                            break

                    memory_lengths[i, j] = new_memory
                else:
                    memory_lengths[i, j] = M_max

                m = memory_lengths[i, j]
                count_now = history_count[i, j]

                if count_now == 0:
                    own_weighted = 0.0
                elif count_now < m:
                    own_weighted = 0.0
                    for idx in range(count_now):
                        own_weighted += payoff_history[i, j, idx]
                    own_weighted /= count_now
                else:
                    own_weighted = 0.0
                    for idx in range(m):
                        own_weighted += payoff_history[i, j, count_now - m + idx] * weight_cache[m, idx]

                neighbor_sum = 0.0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    m_n = memory_lengths[ni, nj]
                    count_n = history_count[ni, nj]

                    if count_n == 0:
                        neighbor_weighted = 0.0
                    elif count_n < m_n:
                        neighbor_weighted = 0.0
                        for idx in range(count_n):
                            neighbor_weighted += payoff_history[ni, nj, idx]
                        neighbor_weighted /= count_n
                    else:
                        neighbor_weighted = 0.0
                        for idx in range(m_n):
                            neighbor_weighted += payoff_history[ni, nj, count_n - m_n + idx] * weight_cache[m_n, idx]

                    neighbor_sum += neighbor_weighted

                avg_neighbor = neighbor_sum / 4.0
                fused_reward = (1.0 - alpha_r) * own_weighted + alpha_r * avg_neighbor

                next_state = 0
                for k in range(4):
                    ni, nj = neighbor_indices[i, j, k]
                    if actions[ni, nj] == 0:
                        next_state += 1

                current_Q = Q_tables[i, j, state, action]
                max_next_Q = max(Q_tables[i, j, next_state, 0], Q_tables[i, j, next_state, 1])
                td_target = fused_reward + gamma * max_next_Q

                td_count = history_count[i, j]
                if td_count < M_max:
                    td_target_history[i, j, td_count] = td_target
                else:
                    for idx in range(M_max - 1):
                        td_target_history[i, j, idx] = td_target_history[i, j, idx + 1]
                    td_target_history[i, j, M_max - 1] = td_target

                td_count_now = min(td_count + 1, M_max)
                if td_count_now >= m:
                    smoothed_target = 0.0
                    for idx in range(m):
                        smoothed_target += td_target_history[i, j, td_count_now - m + idx] * weight_cache[m, idx]
                else:
                    smoothed_target = td_target

                Q_tables[i, j, state, action] = (1.0 - alpha) * current_Q + alpha * smoothed_target

    def get_cooperation_frequency(self):
        return np.mean(self.strategies == 0)

    def get_q_statistics(self):
        """获取Q值统计"""
        Q_coop = self.Q_tables[:, :, :, 0]
        Q_defect = self.Q_tables[:, :, :, 1]
        return {
            'q_diff': np.mean(Q_coop - Q_defect),
            'q_coop_mean': np.mean(Q_coop),
            'q_defect_mean': np.mean(Q_defect),
            'q_abs_mean': np.mean(np.abs(self.Q_tables))
        }


# ==================== 🎨 参数扫描与可视化 ====================

def parameter_scan_b_values():
    """
    参数扫描：不同背叛诱惑 b 的影响
    包含合作率和Q值演化
    """

    print("=" * 80)
    print("🎨 Parameter Scan: Temptation to Defect (b)")
    print("=" * 80)

    # 参数设置
    L = 100
    alpha_r = 0.5  # 固定信息共享强度
    psi = 1.1
    M_max = 10
    q_init_std = 0.01
    total_steps = 60000

    # 背叛诱惑强度范围
    b_values = [1.5, 1.6]

    # 检查点（记录数据的步数）
    checkpoints = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000,
                   12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

    print(f"\n【Parameters】")
    print(f"  Grid size L:              {L}")
    print(f"  Information sharing α_r:  {alpha_r}")
    print(f"  Adaptive threshold ψ:     {psi}")
    print(f"  Maximum memory M_max:     {M_max}")
    print(f"  Total steps:              {total_steps}")
    print(f"  Temptation b values:      {b_values}")
    print("=" * 80)

    # 存储结果
    results = {}

    # 🎨 颜色方案
    colors = plt.cm.viridis(np.linspace(0, 1, len(b_values)))

    # Numba预热
    print("\n【Numba Warm-up】Compiling...")
    dummy_model = DeepFusionQLearning(L=10, M_max=M_max, b=1.3)
    dummy_model.step()
    print("  ✓ Compilation complete\n")

    # 遍历每个b值
    total_start = time.time()

    for idx, b in enumerate(b_values):
        print(f"{'─' * 80}")
        print(f"Running: b = {b:.1f} ({idx + 1}/{len(b_values)})")
        print(f"{'─' * 80}")

        # 创建模型
        model = DeepFusionQLearning(
            L=L, b=b, alpha_r=alpha_r, psi=psi,
            M_max=M_max, q_init_std=q_init_std
        )

        # 记录数据
        steps_list = [0]
        coop_list = [model.get_cooperation_frequency()]
        q_stats = model.get_q_statistics()
        q_diff_list = [q_stats['q_diff']]
        q_abs_list = [q_stats['q_abs_mean']]

        current_step = 0

        # 打印表头
        print(f"{'Step':<10} {'Cooperation':<15} {'Q_diff':<15} {'Avg|Q|':<15} {'Time(s)':<10}")
        print(f"{'-' * 65}")

        # 初始状态
        print(f"{0:<10} {coop_list[0]:<15.4f} {q_diff_list[0]:<15.4f} {q_abs_list[0]:<15.4f} {'0.00':<10}")

        for checkpoint in checkpoints:
            steps_to_run = checkpoint - current_step

            step_start = time.time()
            for _ in range(steps_to_run):
                model.step()
            step_time = time.time() - step_start

            current_step = checkpoint
            coop = model.get_cooperation_frequency()
            q_stats = model.get_q_statistics()

            steps_list.append(current_step)
            coop_list.append(coop)
            q_diff_list.append(q_stats['q_diff'])
            q_abs_list.append(q_stats['q_abs_mean'])

            print(f"{current_step:<10} {coop:<15.4f} {q_stats['q_diff']:<15.4f} "
                  f"{q_stats['q_abs_mean']:<15.4f} {step_time:<10.2f}")

        # 保存结果
        results[b] = {
            'steps': np.array(steps_list),
            'cooperation': np.array(coop_list),
            'q_diff': np.array(q_diff_list),
            'q_abs': np.array(q_abs_list),
            'color': colors[idx]
        }

        print()

    total_time = time.time() - total_start

    print("=" * 80)
    print(f"✓ All simulations completed in {total_time:.1f} seconds")
    print("=" * 80)

    # ==================== 🎨 绘图（双子图）====================

    print("\n【Visualization】Creating plots...")

    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

    # ===== 子图1: 合作率演化 =====
    for b in b_values:
        data = results[b]
        ax1.plot(data['steps'], data['cooperation'],
                 color=data['color'],
                 linewidth=2.5,
                 marker='o',
                 markersize=5,
                 label=f'b = {b:.1f}',
                 alpha=0.9)

    ax1.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cooperation Frequency', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Evolution of Cooperation Frequency',
                  fontsize=14, fontweight='bold', loc='left', pad=15)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax1.tick_params(axis='both', which='major', labelsize=11)

    # ===== 子图2: Q值差异演化 =====
    for b in b_values:
        data = results[b]
        ax2.plot(data['steps'], data['q_diff'],
                 color=data['color'],
                 linewidth=2.5,
                 marker='s',
                 markersize=5,
                 label=f'b = {b:.1f}',
                 alpha=0.9)

    ax2.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Q-value Difference (Coop - Defect)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Evolution of Q-value Difference',
                  fontsize=14, fontweight='bold', loc='left', pad=15)
    ax2.set_xlim(0, total_steps)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax2.tick_params(axis='both', which='major', labelsize=11)

    # 总标题
    fig.suptitle(f'Impact of Temptation to Defect (b) on Cooperation and Q-values\n' +
                 f'(L={L}, α_r={alpha_r}, M_max={M_max}, ψ={psi})',
                 fontsize=15, fontweight='bold', y=0.995)

    # 紧凑布局
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cooperation_and_qvalue_b_values_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Figure saved: {filename}")

    plt.show()

    # ==================== 📊 保存数据 ====================

    print("\n【Data Export】Saving numerical data...")

    # 保存为CSV
    data_filename = f"evolution_data_b_values_{timestamp}.csv"

    with open(data_filename, 'w') as f:
        # 写入表头
        header = "Step"
        for b in b_values:
            header += f",Coop_b{b:.1f},Qdiff_b{b:.1f},Qabs_b{b:.1f}"
        f.write(header + "\n")

        # 写入数据
        for i, step in enumerate(results[b_values[0]]['steps']):
            row = f"{step}"
            for b in b_values:
                coop = results[b]['cooperation'][i]
                q_diff = results[b]['q_diff'][i]
                q_abs = results[b]['q_abs'][i]
                row += f",{coop:.6f},{q_diff:.6f},{q_abs:.6f}"
            f.write(row + "\n")

    print(f"  ✓ Data saved: {data_filename}")

    # ==================== 📈 统计分析 ====================

    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)

    print(f"\n{'b':<6} {'Coop_Init':<12} {'Coop_Final':<12} {'Change':<10} "
          f"{'Q_diff_Init':<12} {'Q_diff_Final':<12}")
    print("-" * 80)

    for b in b_values:
        data = results[b]
        coop_init = data['cooperation'][0]
        coop_final = data['cooperation'][-1]
        coop_change = coop_final - coop_init
        q_diff_init = data['q_diff'][0]
        q_diff_final = data['q_diff'][-1]

        print(f"{b:<6.1f} {coop_init:<12.4f} {coop_final:<12.4f} {coop_change:+<10.4f} "
              f"{q_diff_init:<12.4f} {q_diff_final:<12.4f}")

    print("=" * 80)
    print("\n🎉 Analysis complete!")

    return results


if __name__ == "__main__":
    results = parameter_scan_b_values()