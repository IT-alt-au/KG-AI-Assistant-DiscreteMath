import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator

# ==========================================
# 1. 载入真实实验数据
# ==========================================
data = {
    1.10: [0.502, 0.592, 0.665, 0.738, 0.794, 0.831, 0.859, 0.891, 0.909, 0.932, 0.943, 0.952, 0.964, 0.968, 0.978],
    1.15: [0.502, 0.592, 0.665, 0.738, 0.794, 0.830, 0.856, 0.886, 0.905, 0.929, 0.938, 0.948, 0.962, 0.965, 0.975],
    1.20: [0.502, 0.592, 0.664, 0.737, 0.787, 0.824, 0.848, 0.880, 0.893, 0.919, 0.933, 0.941, 0.955, 0.960, 0.968],
    1.25: [0.502, 0.591, 0.663, 0.735, 0.777, 0.799, 0.834, 0.864, 0.853, 0.876, 0.904, 0.906, 0.914, 0.917, 0.924],
    1.30: [0.502, 0.591, 0.662, 0.730, 0.735, 0.746, 0.776, 0.791, 0.763, 0.804, 0.805, 0.808, 0.801, 0.809, 0.846],
    1.35: [0.502, 0.590, 0.661, 0.711, 0.682, 0.700, 0.726, 0.724, 0.715, 0.757, 0.749, 0.729, 0.736, 0.732, 0.749],
    1.40: [0.502, 0.590, 0.657, 0.664, 0.642, 0.658, 0.682, 0.648, 0.684, 0.657, 0.676, 0.664, 0.673, 0.645, 0.664],
    1.45: [0.502, 0.589, 0.638, 0.603, 0.639, 0.599, 0.661, 0.611, 0.639, 0.600, 0.654, 0.613, 0.638, 0.596, 0.604],
    1.50: [0.502, 0.589, 0.596, 0.564, 0.605, 0.553, 0.622, 0.557, 0.608, 0.569, 0.592, 0.579, 0.553, 0.540, 0.532]
}

steps = np.arange(200, 3200, 200)


# ==========================================
# 2. EMA 平滑函数
# ==========================================
def smooth_ema(scalars, weight=0.65):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


# ==========================================
# 3. 绘图设置
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
fig, ax = plt.subplots(figsize=(7, 4.8))

colors = cm.viridis(np.linspace(0.1, 0.95, len(data)))

for idx, (b_val, raw_coop) in enumerate(data.items()):
    smoothed_coop = smooth_ema(raw_coop, weight=0.65)

    x_new = np.linspace(steps.min(), steps.max(), 300)
    pchip = PchipInterpolator(steps, smoothed_coop)
    y_smooth_curve = pchip(x_new)

    lw = 2.5 if b_val in [1.10, 1.30, 1.50] else 1.5
    alpha = 1.0 if b_val in [1.10, 1.30, 1.50] else 0.7
    ax.plot(x_new, y_smooth_curve, color=colors[idx], lw=lw, alpha=alpha, label=f'$b = {b_val:.2f}$')

# ==========================================
# 4. 图表修饰
# ==========================================
ax.axhline(0.5, color='gray', ls='--', lw=1.5, alpha=0.8)
ax.grid(True, linestyle=':', alpha=0.6)

ax.set_xlabel('Time Steps ($t$)', fontweight='bold', fontsize=13)
ax.set_ylabel('Cooperation Rate ($\\rho_C$)', fontweight='bold', fontsize=13)

ax.set_xlim(0, 3100)
ax.set_ylim(0.4, 1.02)

ax.legend(loc='lower right', ncol=2, framealpha=0.9, edgecolor='black',
          fontsize=9, borderpad=0.5, labelspacing=0.3, handlelength=1.5)

plt.tight_layout()

# 重点修改在这里：改成了超清的 PNG 格式
output_filename = 'mappo_ctde_evolution_final.png'
plt.savefig(output_filename, format='png', dpi=600, bbox_inches='tight')
plt.close()

print(f"🎉 终极版超清 PNG 已生成: {output_filename}")
