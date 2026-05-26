import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit  # 引入Numba加速（关键！）

# --------------------------
# 1. 核心参数
# --------------------------
L = 50  # 50×50网格
total_steps = 50000  # 总步数
stat_steps = 2000  # 最后2000步统计稳态
num_runs = 10  # 独立运行10次
alpha = 0.1
gamma = 0.9
epsilon = 0.02
b_values = np.linspace(1.0, 2.0, 10)  # 10个b值
alpha_r_list = [0.0,  0.1,0.2, 0.3,0.4,0.5, 0.6,0.7, 0.8, 0.9, 1.0]
payoff_matrix = np.array([[1, 0], [0, 0]])
fermi_K = 0.1


# --------------------------
# 2. 用Numba加速核心函数
# --------------------------
@njit  # 编译为机器码，加速10~100倍
def get_neighbors_numba(i, j, L):
    return [
        ((i - 1) % L, j),
        ((i + 1) % L, j),
        (i, (j - 1) % L),
        (i, (j + 1) % L)
    ]


@njit
def simulate_q_learning_numba(b, alpha_r, L, total_steps, stat_steps, alpha, gamma, epsilon):
    strategies = np.random.randint(0, 2, size=(L, L))
    Q = np.zeros((L, L, 5, 2))  # [i,j,state,action]
    current_payoff = np.zeros((L, L))
    coop_counts = 0.0
    count = 0  # 统计步数计数器

    for step in range(total_steps):
        # 1. 计算收益
        new_payoff = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                a_i = strategies[i, j]
                pay = 0.0
                neighbors = get_neighbors_numba(i, j, L)
                for (x, y) in neighbors:
                    a_j = strategies[x, y]
                    if a_i == 1 and a_j == 0:
                        pay += b
                    else:
                        pay += payoff_matrix[a_i, a_j]
                new_payoff[i, j] = pay / 4
        current_payoff = new_payoff

        # 2. 更新策略和Q表
        new_strategies = strategies.copy()
        for i in range(L):
            for j in range(L):
                # 计算状态s（邻居中合作者数量）
                neighbors = get_neighbors_numba(i, j, L)
                s = 0
                for (x, y) in neighbors:
                    if strategies[x, y] == 0:
                        s += 1
                # 邻居平均收益
                P_bar = 0.0
                for (x, y) in neighbors:
                    P_bar += current_payoff[x, y]
                P_bar /= 4
                # 奖励r
                r = (1 - alpha_r) * current_payoff[i, j] + alpha_r * P_bar
                # 下一状态s'
                s_prime = 0
                for (x, y) in neighbors:
                    if new_strategies[x, y] == 0:
                        s_prime += 1
                # 更新Q值
                current_a = strategies[i, j]
                min_next_Q = np.min(Q[i, j, s_prime, :])
                Q[i, j, s, current_a] = (1 - alpha) * Q[i, j, s, current_a] + alpha * (r + gamma * min_next_Q)
                # ε-贪婪选择动作
                if np.random.rand() < epsilon:
                    new_a = np.random.randint(0, 2)
                else:
                    new_a = np.argmax(Q[i, j, s, :])
                new_strategies[i, j] = new_a
        strategies = new_strategies

        # 3. 统计稳态合作频率
        if step >= total_steps - stat_steps:
            coop_counts += np.mean(strategies == 0)
            count += 1

    return coop_counts / count  # 平均合作频率


# 费米规则模拟（同样用Numba加速）
@njit
def simulate_fermi_numba(b, L, total_steps, stat_steps):
    strategies = np.random.randint(0, 2, size=(L, L))
    coop_counts = 0.0
    count = 0
    for step in range(total_steps):
        # 计算收益
        payoff = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                a_i = strategies[i, j]
                pay = 0.0
                neighbors = get_neighbors_numba(i, j, L)
                for (x, y) in neighbors:
                    a_j = strategies[x, y]
                    if a_i == 1 and a_j == 0:
                        pay += b
                    else:
                        pay += payoff_matrix[a_i, a_j]
                payoff[i, j] = pay / 4
        # 费米规则更新
        new_strategies = strategies.copy()
        for i in range(L):
            for j in range(L):
                x, y = get_neighbors_numba(i, j, L)[np.random.randint(4)]
                pi_i = payoff[i, j]
                pi_j = payoff[x, y]
                prob = 1.0 / (1.0 + np.exp((pi_i - pi_j) / fermi_K))
                if np.random.rand() < prob:
                    new_strategies[i, j] = strategies[x, y]
        strategies = new_strategies
        # 统计
        if step >= total_steps - stat_steps:
            coop_counts += np.mean(strategies == 0)
            count += 1
    return coop_counts / count


# --------------------------
# 3. 主程序（并行运行）
# --------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    from functools import partial


    # 多进程加速（利用CPU多核）
    def run_parallel(func, params_list):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(func, params_list), total=len(params_list)))
        return results


    # 1. 运行费米规则（并行处理10个b值）
    fermi_func = partial(
        simulate_fermi_numba,
        L=L, total_steps=total_steps, stat_steps=stat_steps
    )
    fermi_results = []
    for b in b_values:
        # 每个b值跑10次独立运行，参数直接传b（标量），不包装成元组
        params = [b for _ in range(num_runs)]  # 关键修正
        runs = run_parallel(fermi_func, params)
        fermi_results.append(np.mean(runs))

    # 2. 运行Q学习（同样修正参数传递）
    q_results = {ar: [] for ar in alpha_r_list}
    for alpha_r in alpha_r_list:
        q_func = partial(
            simulate_q_learning_numba,
            alpha_r=alpha_r, L=L, total_steps=total_steps,
            stat_steps=stat_steps, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        for b in b_values:
            params = [b for _ in range(num_runs)]  # 关键修正
            runs = run_parallel(q_func, params)
            q_results[alpha_r].append(np.mean(runs))

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    for alpha_r in alpha_r_list:
        plt.plot(b_values, q_results[alpha_r], label=f'α_r={alpha_r}')
    plt.plot(b_values, fermi_results, 'orange', linestyle='--', label='Fermi rule')
    plt.xlabel('Temptation to defect (b)')
    plt.ylabel('Cooperation frequency (f_C)')
    plt.title('Cooperation frequency vs b (Simplified Parameters)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig1_.png', dpi=300)
    plt.show()