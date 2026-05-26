import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import multiprocessing as mp
from functools import partial

# --------------------------
# 1. 核心参数
# --------------------------
L = 50  # 网格大小
total_steps = 200000  # 延长时间步数
num_runs = 10  # 独立运行次数（平衡计算速度与波动）
b = 1.6  # 固定背叛诱惑值
# alpha_r 细化为 0.0, 0.1, ..., 1.0（共11个值）
alpha_r_list = np.round(np.linspace(0, 1, 11), 1).tolist()
alpha = 0.1  # Q-learning 学习率
gamma = 0.9  # Q-learning 折扣因子
epsilon = 0.02  # ε-贪婪探索概率
payoff_matrix = np.array([[1, 0], [0, 0]], dtype=np.float32)


# --------------------------
# 2. Numba 加速函数
# --------------------------
@njit
def get_neighbors(i, j, L):
    return [
        ((i - 1) % L, j), ((i + 1) % L, j),
        (i, (j - 1) % L), (i, (j + 1) % L)
    ]


@njit
def calculate_payoff(strategies, L, b, payoff_matrix):
    payoff = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            a_i = strategies[i, j]
            total_pay = 0.0
            for (x, y) in get_neighbors(i, j, L):
                a_j = strategies[x, y]
                if a_i == 1 and a_j == 0:
                    total_pay += b  # 背叛者从合作者处获得 b
                else:
                    total_pay += payoff_matrix[a_i, a_j]
            payoff[i, j] = total_pay / 4  # 4 个邻居平均
    return payoff


@njit
def count_cooperators(neighbors, strategies):
    count = 0
    for (x, y) in neighbors:
        if strategies[x, y] == 0:  # 0 表示合作策略
            count += 1
    return count


@njit
def update_strategies(strategies, Q, payoff, L, alpha_r, alpha, gamma, epsilon):
    new_strategies = strategies.copy()
    for i in range(L):
        for j in range(L):
            neighbors = get_neighbors(i, j, L)
            # 计算当前状态 s（邻居中合作者数量）
            s = count_cooperators(neighbors, strategies)

            # 计算邻居平均收益 P_bar
            P_bar = 0.0
            for (x, y) in neighbors:
                P_bar += payoff[x, y]
            P_bar /= 4  # 4 个邻居平均

            # 总奖励 r（混合自身收益与邻居平均收益）
            r = (1 - alpha_r) * payoff[i, j] + alpha_r * P_bar

            # 计算下一状态 s'（基于新策略的邻居）
            s_prime = count_cooperators(neighbors, new_strategies)

            # 更新 Q 值（Q-learning 核心）
            current_a = strategies[i, j]
            min_next_Q = np.min(Q[i, j, s_prime, :])
            Q[i, j, s, current_a] = (1 - alpha) * Q[i, j, s, current_a] + alpha * (r + gamma * min_next_Q)

            # ε-贪婪选择新策略
            if np.random.rand() < epsilon:
                new_a = np.random.randint(0, 2)
            else:
                new_a = np.argmax(Q[i, j, s, :])
            new_strategies[i, j] = new_a
    return new_strategies, Q


@njit
def numba_main_loop(strategies, Q, L, total_steps, b, alpha_r, alpha, gamma, epsilon, payoff_matrix):
    coop_history = np.zeros(total_steps)  # 记录每代合作频率
    for step in range(total_steps):
        # 1. 计算所有智能体的收益
        payoff = calculate_payoff(strategies, L, b, payoff_matrix)

        # 2. 更新策略和 Q 表
        strategies, Q = update_strategies(strategies, Q, payoff, L, alpha_r, alpha, gamma, epsilon)

        # 3. 记录当前代合作频率
        coop_history[step] = np.mean(strategies == 0)  # 0 表示合作
    return coop_history


# --------------------------
# 3. 模拟主函数（多进程并行）
# --------------------------
def simulate_dynamics(alpha_r):
    # 初始化：随机策略（0=合作，1=背叛） + 全零 Q 表
    strategies = np.random.randint(0, 2, size=(L, L))
    Q = np.zeros((L, L, 5, 2), dtype=np.float32)  # s 范围：0~4（4个邻居），a 范围：0~1

    # 调用 Numba 加速的主循环
    return numba_main_loop(strategies, Q, L, total_steps, b, alpha_r, alpha, gamma, epsilon, payoff_matrix)


def run_simulation(alpha_r):
    # 多次独立运行，取平均以平滑波动
    all_histories = []
    for _ in range(num_runs):
        history = simulate_dynamics(alpha_r)
        all_histories.append(history)
    return np.mean(all_histories, axis=0)  # 平均多条轨迹


# --------------------------
# 4. 运行模拟 + 绘图（复现原图趋势）
# --------------------------
if __name__ == "__main__":
    # 多进程并行（加速 11 个 alpha_r 的模拟）
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(run_simulation, alpha_r_list),
            total=len(alpha_r_list),
            desc="Simulating dynamics"
        ))

    # 整理结果：alpha_r -> 合作频率历史
    history_dict = {ar: res for ar, res in zip(alpha_r_list, results)}

    # 绘图：匹配原图的趋势（初期波动、后期稳定）
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(total_steps)

    # 绘制初始合作频率参考线（f0=0.5）
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Initial frequency (f0=0.5)')

    # 按 alpha_r 从大到小绘图（让高 alpha_r 的曲线在上方更清晰）
    for alpha_r in sorted(alpha_r_list, reverse=True):
        plt.plot(time_steps, history_dict[alpha_r], label=f'α_r={alpha_r}')

    # 添加参数说明文本框
    param_text = (
        f"Grid size (L) = {L}\n"
        f"Total steps = {total_steps}\n"
        f"Number of runs = {num_runs}\n"
        f"b = {b}\n"
        f"α (learning rate) = {alpha}\n"
        f"γ (discount factor) = {gamma}\n"
        f"ε (exploration) = {epsilon}"
    )
    # 将文本框放在图表外右下角，bbox_to_anchor 调整位置
    plt.text(
        1.02, 0.02, param_text,
        fontsize=9,
        ha='left', va='bottom',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.9, pad=5)
    )

    plt.xlabel('Time (generations)')
    plt.ylabel('Cooperation frequency (f_C)')
    plt.title(f'Fig. 2: f_C vs time (b={b})')
    plt.ylim(0, 1.05)  # 限制 y 轴范围，突出波动
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例放右侧，避免遮挡曲线
    plt.grid(alpha=0.3)
    plt.tight_layout()  # 自动调整布局（避开文本框）
    plt.show()