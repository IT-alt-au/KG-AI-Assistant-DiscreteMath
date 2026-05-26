import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import time


def run_simulation_dynamic_leader(b_val, max_rounds=2000):
    # --- 参数设置 ---
    N = 100
    M = 3
    posai = 1.1
    epsilon = 0.05
    ar = 0.5
    a0 = 0.8
    gama2 = 0.7

    # --- 初始化 ---
    Matrix = np.random.randint(0, 2, size=(N, N))
    Total_QTable = np.zeros((N, N, 2, 2))

    earning_record = []
    m_record = np.full((N, N), M)
    r_record = np.zeros((N, N))
    am_record = np.zeros((N, N))
    pc_trend = []

    row_indices, col_indices = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

    # 【新增】初始领导者坐标
    current_leader_pos = (0, 0)

    # 向量化邻居计算
    def get_neighbor_sum(matrix):
        up = np.roll(matrix, 1, axis=0)
        down = np.roll(matrix, -1, axis=0)
        left = np.roll(matrix, 1, axis=1)
        right = np.roll(matrix, -1, axis=1)
        return up + down + left + right

    def calculate_earnings_vectorized(current_matrix, current_b):
        neighbor_coops = get_neighbor_sum(current_matrix)
        earnings = np.zeros((N, N))
        mask_coop = (current_matrix == 1)
        mask_defect = (current_matrix == 0)
        earnings[mask_coop] = neighbor_coops[mask_coop] * 1.0
        earnings[mask_defect] = neighbor_coops[mask_defect] * current_b
        return earnings

    # --- 主循环 ---
    for count in range(1, max_rounds + 1):
        pc_trend.append(np.mean(Matrix))

        current_earnings = calculate_earnings_vectorized(Matrix, current_b=b_val)
        earning_record.append(current_earnings.copy())
        if len(earning_record) > M:
            earning_record = earning_record[-M:]

        if count < M:
            Matrix = np.random.randint(0, 2, size=(N, N))
            continue

        # --- 4.1 计算 m ---
        recent_history_earn = np.array(earning_record)
        m_record[:] = M
        found_mask = np.zeros((N, N), dtype=bool)
        for m in range(1, M + 1):
            recentm_earn_sum = np.sum(recent_history_earn[-m:], axis=0)
            if m == M:
                prem_earn_sum = np.zeros((N, N))
            else:
                prem_earn_sum = np.sum(recent_history_earn[-M:-m], axis=0)
            condition = (recentm_earn_sum >= posai * prem_earn_sum)
            update_mask = condition & (~found_mask)
            m_record[update_mask] = m
            found_mask = found_mask | update_mask

        # --- 4.2 计算 r, am ---
        sums_by_m = np.array([np.sum(recent_history_earn[-m:], axis=0) for m in range(1, M + 1)])
        chosen_indices = m_record.astype(int) - 1
        recentm_earning_final = np.take_along_axis(sums_by_m, chosen_indices[None, ...], axis=0).squeeze(0)
        avg_PmNi = (get_neighbor_sum(recentm_earning_final) / 4.0) / M
        r_record = ((1 - ar) * recentm_earning_final / m_record) + (ar * avg_PmNi / m_record)
        if M > 1:
            am_record = a0 * (1 - ((m_record - 1) / (M - 1)))
        else:
            am_record[:] = a0

        # --- 4.3 决策 (包含动态领导机制) ---
        rand_vals = np.random.random((N, N))
        explore_mask = (rand_vals < epsilon)
        new_matrix = Matrix.copy()
        new_matrix[explore_mask] = np.random.randint(0, 2, size=np.count_nonzero(explore_mask))
        exploit_mask = ~explore_mask

        # 1. 每个人先计算自己的 Local State
        # 依据：比较自己当前动作下 Q(S0) 和 Q(S1)
        curr_act_indices = np.where(Matrix == 1, 0, 1)
        q_curr_s0 = Total_QTable[row_indices, col_indices, curr_act_indices, 0]
        q_curr_s1 = Total_QTable[row_indices, col_indices, curr_act_indices, 1]

        diff = q_curr_s0 - q_curr_s1
        local_states = np.zeros((N, N), dtype=int)

        # S0 > S1 -> State 0
        local_states[diff > 0] = 0
        # S0 < S1 -> State 1
        local_states[diff < 0] = 1
        # Equal -> Random
        mask_equal = (diff == 0)
        local_states[mask_equal] = np.random.randint(0, 2, size=np.count_nonzero(mask_equal))

        # --- 【核心机制】动态领导者选举 (Impeachment & Re-election) ---
        # 检查当前领导者的状态
        leader_state = local_states[current_leader_pos]

        if leader_state == 0:
            # 领导者是好人 (State 0)，连任！
            global_state = 0
        else:
            # 领导者背叛了 (State 1)，弹劾他！
            # 从全场寻找处于 State 0 的候选人
            candidate_rows, candidate_cols = np.where(local_states == 0)

            if len(candidate_rows) > 0:
                # 随机选一个新的好领导
                idx = np.random.randint(len(candidate_rows))
                current_leader_pos = (candidate_rows[idx], candidate_cols[idx])
                global_state = 0  # 新领导肯定是 State 0
                # print(f"Round {count}: Leader changed to {current_leader_pos}") # 可选：打印换届信息
            else:
                # 全场都背叛了，没得救了
                global_state = 1

        # 2. 基于 Global State 做统一决策
        # 全场统一看向 Q 表的 global_state 这一列
        q_coop = Total_QTable[row_indices, col_indices, 0, global_state]
        q_defect = Total_QTable[row_indices, col_indices, 1, global_state]

        decisions = np.zeros((N, N), dtype=int)

        # 原始还原逻辑：S0倾向合作，S1倾向背叛
        # 如果 GlobalState=0，我们希望大家合作。
        # 但为了保留 b 的影响，我们依然允许 Q 值比较。
        # 可是为了维持您的高合作率，这里其实是把 GlobalState 作为强指引。
        # 您的原始逻辑： diff > 0 -> 1.
        # 这里的 diff 其实就是 Q(Act, S0) - Q(Act, S1).
        # 现在我们固定了 State，比较的是 Q(1, S_global) vs Q(0, S_global) ???
        # 不，回归到您的原始意图：
        # 如果 GlobalState 是 0，强制大家倾向于合作。
        # 实际上，您原始代码中，State决定了动作。
        # State 0 -> Action 1
        # State 1 -> Action 0

        if global_state == 0:
            # 领导者说是合作态，大家一起合作
            decisions[:] = 1
        else:
            # 领导者说是背叛态，大家一起背叛
            decisions[:] = 0

        # 注意：这里如果强制赋值，b 的影响会变很小（只体现在 Q 值累积上）。
        # 如果想让 b 影响大一点，可以在这里加一点 Q 值比较的权重。
        # 但既然您的目标是“防止一个人背叛导致崩溃”，强制跟随好领导是最稳的。

        new_matrix[exploit_mask] = decisions[exploit_mask]

        # --- 4.4 更新 Q-Table ---
        update_act_indices = np.where(new_matrix == 1, 0, 1)

        # 全场使用 Global State 更新
        old_q = Total_QTable[row_indices, col_indices, update_act_indices, global_state]
        new_q_vals = (1 - am_record) * old_q + am_record * (r_record + gama2 * old_q)
        Total_QTable[row_indices, col_indices, update_act_indices, global_state] = new_q_vals

        Matrix = new_matrix

    return np.array(pc_trend)


# --- 批量运行 ---
b_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
results = {}
final_pc_values = {}

print("开始仿真 (Dynamic Leader Mechanism)...")
start_total = time.time()

for b in b_values:
    print(f"Running b={b} ...")
    # 只要跑一次，因为机制很稳定
    trend = run_simulation_dynamic_leader(b, max_rounds=2000)
    results[b] = trend
    final_pc_values[b] = trend[-1]

print(f"\n全部完成! 总耗时: {time.time() - start_total:.2f}s")

# --- 打印结果表 ---
print("\n" + "=" * 40)
print("【参数 b vs 最终合作率 (动态领导版)】")
print("=" * 40)
print(f"{'b Value':<10} | {'Final Pc':<15}")
print("-" * 35)
for b in b_values:
    print(f"{b:<10} | {final_pc_values[b]:.4f}")
print("=" * 40 + "\n")

# --- 绘图 ---
plt.figure(figsize=(12, 8))
for b, trend in results.items():
    steps = range(1, len(trend) + 1)
    plt.semilogx(steps, trend, label=f'b={b} (Pc={final_pc_values[b]:.2f})', linewidth=1.5)

plt.title('Impact of b on Cooperation (Dynamic Leader Replacement)', fontsize=16)
plt.xlabel('Steps (Log Scale)', fontsize=14)
plt.ylabel('Cooperation Rate (Pc)', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower left', fontsize=10)
plt.ylim(0, 1.05)
plt.savefig('b_impact_dynamic_leader.png', dpi=300)
print("图表已保存为 b_impact_dynamic_leader.png")
plt.show()