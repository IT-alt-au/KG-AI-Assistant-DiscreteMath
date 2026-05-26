import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import multiprocessing as mp

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DeepFusionQLearning:


    def __init__(self, L=100, b=1.6, alpha=0.1, gamma=0.9, epsilon=0.02,
                 alpha_r=0.5, psi=1.1, M_max=10, q_init_std=0.01):

        self.L = L
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_r = alpha_r
        self.psi = psi
        self.M_max = M_max

        # 收益矩阵
        self.payoff_matrix = np.array([[1.0, 0.0], [b, 0.0]])

        # 初始化
        self.strategies = np.random.randint(0, 2, (L, L))
        self.Q_tables = np.random.randn(L, L, 5, 2) * q_init_std
        self.memory_lengths = np.full((L, L), M_max, dtype=int)

        # 历史记录
        self.payoff_history = [[deque(maxlen=M_max) for _ in range(L)]
                               for _ in range(L)]
        self.td_target_history = [[deque(maxlen=M_max) for _ in range(L)]
                                  for _ in range(L)]

        # 🔥 优化：预计算邻居索引（减少模运算）
        self.neighbors = self._precompute_neighbors()

    def _precompute_neighbors(self):
        """预计算所有位置的邻居索引"""
        neighbors = np.zeros((self.L, self.L, 4, 2), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                neighbors[i, j, 0] = [(i - 1) % self.L, j]
                neighbors[i, j, 1] = [(i + 1) % self.L, j]
                neighbors[i, j, 2] = [i, (j - 1) % self.L]
                neighbors[i, j, 3] = [i, (j + 1) % self.L]
        return neighbors

    def compute_payoff(self, i, j):
        """计算收益（算法不变）"""
        strategy = self.strategies[i, j]
        payoff = 0.0
        for k in range(4):
            ni, nj = self.neighbors[i, j, k]
            payoff += self.payoff_matrix[strategy, self.strategies[ni, nj]]
        return payoff

    def update_memory_length(self, i, j):
        """自适应记忆长度（算法不变）"""
        history = self.payoff_history[i][j]
        if len(history) < 2:
            return self.M_max

        history_array = np.array(history)
        for m_tilde in range(1, len(history)):
            recent_sum = np.sum(history_array[-m_tilde:])
            past_sum = np.sum(history_array[:-m_tilde])
            if past_sum > 0 and recent_sum >= self.psi * past_sum:
                return m_tilde
        return self.M_max

    def compute_memory_weighted_payoff(self, i, j):
        """记忆加权收益（算法不变）"""
        m = self.memory_lengths[i, j]
        history = self.payoff_history[i][j]

        if len(history) == 0:
            return 0.0
        if len(history) < m:
            return np.mean(history)

        recent_payoffs = np.array(list(history)[-m:])
        weights = np.exp(np.linspace(-1, 0, m))
        weights /= weights.sum()
        return np.sum(recent_payoffs * weights)

    def compute_fused_reward(self, i, j):
        """融合奖励（算法不变）"""
        own_weighted = self.compute_memory_weighted_payoff(i, j)

        neighbor_sum = 0.0
        for k in range(4):
            ni, nj = self.neighbors[i, j, k]
            neighbor_sum += self.compute_memory_weighted_payoff(ni, nj)

        avg_neighbor = neighbor_sum / 4.0
        return (1 - self.alpha_r) * own_weighted + self.alpha_r * avg_neighbor

    def get_state(self, i, j):
        """状态计算（算法不变）"""
        coop_count = 0
        for k in range(4):
            ni, nj = self.neighbors[i, j, k]
            if self.strategies[ni, nj] == 0:
                coop_count += 1
        return coop_count

    def select_action(self, i, j, state):
        """动作选择（算法不变）"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)

        Q_values = self.Q_tables[i, j, state]
        max_Q = np.max(Q_values)
        max_actions = np.where(Q_values == max_Q)[0]
        return np.random.choice(max_actions)

    def update_Q_with_memory_smoothing(self, i, j, state, action,
                                       reward, next_state):
        """Q表更新（算法不变）"""
        m = self.memory_lengths[i, j]
        current_Q = self.Q_tables[i, j, state, action]
        max_next_Q = np.max(self.Q_tables[i, j, next_state])
        td_target = reward + self.gamma * max_next_Q

        self.td_target_history[i][j].append(td_target)
        td_history = self.td_target_history[i][j]

        if len(td_history) >= m:
            recent_targets = np.array(list(td_history)[-m:])
            weights = np.exp(np.linspace(-1, 0, m))
            weights /= weights.sum()
            smoothed_target = np.sum(recent_targets * weights)
        else:
            smoothed_target = td_target

        self.Q_tables[i, j, state, action] = \
            (1 - self.alpha) * current_Q + self.alpha * smoothed_target

    def step(self):
        """单步演化（算法不变）"""
        new_strategies = self.strategies.copy()

        for i in range(self.L):
            for j in range(self.L):
                state = self.get_state(i, j)
                action = self.select_action(i, j, state)

                current_payoff = self.compute_payoff(i, j)
                self.payoff_history[i][j].append(current_payoff)

                self.memory_lengths[i, j] = self.update_memory_length(i, j)
                fused_reward = self.compute_fused_reward(i, j)

                new_strategies[i, j] = action

                old_strategy = self.strategies[i, j]
                self.strategies[i, j] = action
                next_state = self.get_state(i, j)
                self.strategies[i, j] = old_strategy

                self.update_Q_with_memory_smoothing(i, j, state, action,
                                                    fused_reward, next_state)

        self.strategies = new_strategies

    def get_cooperation_frequency(self):
        """计算合作率"""
        return np.mean(self.strategies == 0)

    def run_simulation(self, total_steps=15000, stat_steps=2000):
        """运行模拟并统计稳态"""
        coop_counts = 0.0
        count = 0

        for step in range(total_steps):
            self.step()

            if step >= total_steps - stat_steps:
                coop_counts += self.get_cooperation_frequency()
                count += 1

        return coop_counts / count if count > 0 else 0.0


# ==================== 单次运行函数 ====================

def run_single_simulation(params):
    """单次模拟"""
    b, alpha_r, L, total_steps, stat_steps, psi, q_init_std = params

    model = DeepFusionQLearning(
        L=L, b=b, alpha_r=alpha_r, psi=psi,
        q_init_std=q_init_std
    )

    return model.run_simulation(total_steps=total_steps, stat_steps=stat_steps)


# ==================== 主函数（带进度条） ====================

def full_parameter_scan_with_repeats():
    """完整参数扫描"""

    # ==================== 参数设置 ====================

    b_values = np.arange(1.0, 1.5, 0.1)
    alpha_r_values = [ 0.7,0.8]

    L = 50
    total_steps = 5000
    stat_steps = 1000
    psi = 1.1
    q_init_std = 0.01
    num_runs = 3

    # ==================== 打印配置 ====================

    print("=" * 80)
    print("🚀 深度融合模型：参数扫描（优化版）")
    print("=" * 80)
    print(f"\n参数: L={L}, steps={total_steps}, stat_steps={stat_steps}")
    print(f"扫描: b={len(b_values)}点, α_r={len(alpha_r_values)}条")
    print(f"重复: 每组{num_runs}次")
    print(f"总任务: {len(b_values) * len(alpha_r_values) * num_runs}")
    print(f"并行: {mp.cpu_count()}核")
    print("=" * 80)

    # ==================== 准备任务 ====================

    all_tasks = []
    task_info = []

    for alpha_r in alpha_r_values:
        for b in b_values:
            for run in range(num_runs):
                task = (b, alpha_r, L, total_steps, stat_steps, psi, q_init_std)
                all_tasks.append(task)
                task_info.append({'b': b, 'alpha_r': alpha_r, 'run': run})

    total_tasks = len(all_tasks)
    print(f"\n开始计算 ({total_tasks} 个任务)...")
    start_time = time.time()

    # ==================== 🔥 并行运行 + 进度条 ====================

    all_results = []
    completed = 0

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.imap_unordered(run_single_simulation, all_tasks):
            all_results.append(result)
            completed += 1

            # 🔥 进度条显示
            if completed % 5 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_tasks - completed)
                progress = completed / total_tasks

                # 进度条
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)

                # 动态信息
                print(f"\r  [{completed:3d}/{total_tasks}] |{bar}| "
                      f"{progress * 100:5.1f}% | "
                      f"已用{elapsed / 60:4.1f}分 | "
                      f"剩余{eta / 60:4.1f}分", end='', flush=True)

    print()  # 换行

    total_time = time.time() - start_time
    print(f"\n✅ 完成！耗时 {total_time / 60:.2f} 分钟")
    print(f"   平均每任务 {total_time / total_tasks:.2f} 秒")

    # ==================== 整理结果 ====================

    results_dict = {}
    for i, result in enumerate(all_results):
        info = task_info[i]
        key = (info['b'], info['alpha_r'])
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(result)

    results = {}
    results_std = {}

    for alpha_r in alpha_r_values:
        results[alpha_r] = []
        results_std[alpha_r] = []

        for b in b_values:
            key = (b, alpha_r)
            runs = results_dict[key]
            results[alpha_r].append(np.mean(runs))
            results_std[alpha_r].append(np.std(runs))

    # ==================== 绘图 ====================

    print("\n绘制图表...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图1：合作率曲线
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(alpha_r_values)))

    for idx, alpha_r in enumerate(alpha_r_values):
        mean_vals = results[alpha_r]
        std_vals = results_std[alpha_r]

        ax1.plot(b_values, mean_vals,
                 marker='o', markersize=6, linewidth=2.5,
                 color=colors[idx],
                 label=f'$\\alpha_r$ = {alpha_r:.1f}',
                 alpha=0.85)

        ax1.fill_between(b_values,
                         np.array(mean_vals) - np.array(std_vals),
                         np.array(mean_vals) + np.array(std_vals),
                         color=colors[idx], alpha=0.2)

    if 0.0 in results:
        ax1.plot(b_values, results[0.0],
                 color='orange', linewidth=4, linestyle='-',
                 label='Fermi', alpha=1.0, zorder=1)

    ax1.set_xlabel('$b$ (Temptation to Defect)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cooperation Frequency', fontsize=14, fontweight='bold')
    ax1.set_xlim([1.0, 2.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_title('(a) Cooperation vs Temptation', fontsize=13, fontweight='bold')

    # 图2：热图
    heatmap_data = np.zeros((len(alpha_r_values), len(b_values)))
    for i, alpha_r in enumerate(alpha_r_values):
        heatmap_data[i, :] = results[alpha_r]

    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                    origin='lower', vmin=0, vmax=1)

    ax2.set_xticks(np.arange(len(b_values)))
    ax2.set_xticklabels([f'{b:.1f}' for b in b_values], fontsize=9)
    ax2.set_yticks(np.arange(len(alpha_r_values)))
    ax2.set_yticklabels([f'{ar:.1f}' for ar in alpha_r_values], fontsize=10)

    ax2.set_xlabel('$b$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('$\\alpha_r$', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Heatmap', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    for i in range(len(alpha_r_values)):
        for j in range(len(b_values)):
            ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=7)

    plt.suptitle(f'Deep Fusion Model (L={L}, steps={total_steps}, runs={num_runs})',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    filename = f'parameter_scan_optimized.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存: {filename}")
    plt.show()

    # ==================== 统计报告 ====================

    print(f"\n{'=' * 80}")
    print("📊 统计报告")
    print(f"{'=' * 80}")
    print(f"{'α_r':<6} {'平均':<10} {'最大':<10} {'最小':<10} {'平均σ':<10}")
    print("-" * 80)

    for alpha_r in alpha_r_values:
        coops = np.array(results[alpha_r])
        stds = np.array(results_std[alpha_r])
        print(f"{alpha_r:<6.1f} {np.mean(coops):<10.3f} {np.max(coops):<10.3f} "
              f"{np.min(coops):<10.3f} {np.mean(stds):<10.4f}")

    print("=" * 80)

    return results, results_std, b_values, alpha_r_values


# ==================== 运行 ====================

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

    results, results_std, b_values, alpha_r_values = full_parameter_scan_with_repeats()

