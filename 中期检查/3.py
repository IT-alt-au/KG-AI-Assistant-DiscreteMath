import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from numba import njit
import os


# --------------------------
# 1. 参数配置（50×50网格 + 20万步）
# --------------------------
class Config:
    L = 50  # 50×50网格
    total_steps = 200000  # 总演化步数
    b = 1.6  # 固定背叛诱惑值
    alpha = 0.1  # Q-learning参数
    gamma = 0.9
    epsilon = 0.02
    payoff_matrix = np.array([[1, 0], [0, 0]], dtype=np.float32)

    # 图3核心参数
    alpha_r_list = [0.0, 0.2, 0.4, 0.6, 0.8]  # 5个αᵣ值（列）
    snapshot_times = [100, 10000, 50000, 200000]  # 4个时间点（行）
    output_dir = "fig3"  # 结果保存目录


os.makedirs(Config.output_dir, exist_ok=True)


# --------------------------
# 2. 核心模拟函数（Numba加速）
# --------------------------
@njit
def get_neighbors(i, j, L):
    """获取4个邻居坐标（周期性边界）"""
    return [((i - 1) % L, j), ((i + 1) % L, j), (i, (j - 1) % L), (i, (j + 1) % L)]


@njit
def calculate_payoff(strategies, L, b, payoff_matrix):
    """计算每个智能体的收益"""
    payoff = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            a_i = strategies[i, j]  # 0=合作，1=背叛
            total_pay = 0.0
            for (x, y) in get_neighbors(i, j, L):
                a_j = strategies[x, y]
                if a_i == 1 and a_j == 0:  # 背叛者从合作者处获b
                    total_pay += b
                else:
                    total_pay += payoff_matrix[a_i, a_j]
            payoff[i, j] = total_pay / 4  # 4个邻居平均
    return payoff


@njit
def count_cooperators(neighbors, strategies):
    """统计邻居中合作者数量（状态s）"""
    count = 0
    for (x, y) in neighbors:
        if strategies[x, y] == 0:
            count += 1
    return count


@njit
def update_strategies(strategies, Q, payoff, L, alpha_r, alpha, gamma, epsilon):
    """Q-learning策略更新"""
    new_strategies = strategies.copy()
    for i in range(L):
        for j in range(L):
            neighbors = get_neighbors(i, j, L)
            s = count_cooperators(neighbors, strategies)  # 当前状态

            # 邻居平均收益P_bar
            P_bar = 0.0
            for (x, y) in neighbors:
                P_bar += payoff[x, y]
            P_bar /= 4

            # 总奖励r
            r = (1 - alpha_r) * payoff[i, j] + alpha_r * P_bar

            # 下一状态s'
            s_prime = count_cooperators(neighbors, new_strategies)

            # 更新Q值
            current_a = strategies[i, j]
            min_next_Q = np.min(Q[i, j, s_prime, :])
            Q[i, j, s, current_a] = (1 - alpha) * Q[i, j, s, current_a] + alpha * (r + gamma * min_next_Q)

            # ε-贪婪选择
            new_a = np.random.randint(0, 2) if np.random.rand() < epsilon else np.argmax(Q[i, j, s, :])
            new_strategies[i, j] = new_a
    return new_strategies, Q


# --------------------------
# 3. 生成策略快照（单进程模式）
# --------------------------
@njit
def simulate_snapshots(alpha_r, L, total_steps, snapshot_times, b, alpha, gamma, epsilon, payoff_matrix):
    """为单个αᵣ值生成指定时间点的快照"""
    strategies = np.random.randint(0, 2, size=(L, L))  # 初始随机策略
    Q = np.zeros((L, L, 5, 2), dtype=np.float32)  # Q表：[i,j,state,action]
    snapshots = {}  # {时间点: 策略矩阵}

    for step in range(total_steps + 1):
        if step in snapshot_times:
            snapshots[step] = strategies.copy()  # 保存当前策略
        if step == total_steps:
            break
        # 计算收益并更新策略
        payoff = calculate_payoff(strategies, L, b, payoff_matrix)
        strategies, Q = update_strategies(strategies, Q, payoff, L, alpha_r, alpha, gamma, epsilon)
    return snapshots


def run_simulation(alpha_r):
    """单进程运行单个αᵣ的模拟"""
    return simulate_snapshots(
        alpha_r=alpha_r,
        L=Config.L,
        total_steps=Config.total_steps,
        snapshot_times=Config.snapshot_times,
        b=Config.b,
        alpha=Config.alpha,
        gamma=Config.gamma,
        epsilon=Config.epsilon,
        payoff_matrix=Config.payoff_matrix
    )


# --------------------------
# 4. 绘制图3（4行×5列快照网格）
# --------------------------
def plot_fig3(snapshots_dict):
    """绘制4行×5列的策略快照网格"""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 5, figure=fig, wspace=0.05, hspace=0.15)
    cmap = plt.cm.colors.ListedColormap(['blue', 'red'])  # 蓝=合作，红=背叛

    # 遍历时间点（行）和αᵣ值（列）
    for row, t in enumerate(Config.snapshot_times):
        for col, alpha_r in enumerate(Config.alpha_r_list):
            strategies = snapshots_dict[alpha_r][t]  # 获取策略矩阵
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(strategies, cmap=cmap, vmin=0, vmax=1)  # 绘制策略分布
            ax.axis('off')  # 关闭坐标轴

            # 添加标签（仅第一行和第一列）
            if row == 0:
                ax.set_title(f'αᵣ={alpha_r}', fontsize=10, pad=5)
            if col == 0:
                ax.text(-6, Config.L // 2, f'T={t}', rotation=90,
                        va='center', ha='center', fontsize=10)

    # 保存高清图像
    plt.savefig(f"{Config.output_dir}/fig3_a-t.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图3已保存至 {Config.output_dir}")


# --------------------------
# 主程序（单进程，稳定运行）
# --------------------------
if __name__ == "__main__":
    # 单进程循环生成所有αᵣ值的快照
    print("生成图3策略演化快照（50×50网格，20万步）...")
    results = []
    for alpha_r in tqdm(Config.alpha_r_list, desc="模拟进度"):
        results.append(run_simulation(alpha_r))

    # 整理结果：{αᵣ: {时间点: 策略矩阵}}
    snapshots_dict = {ar: res for ar, res in zip(Config.alpha_r_list, results)}

    # 绘制图3
    plot_fig3(snapshots_dict)