# 空间囚徒困境与多智能体强化学习探索 (Spatial Prisoner's Dilemma MARL)

本项目探索了在空间网格上的囚徒困境游戏（Spatial Prisoner's Dilemma, SPD）中，多智能体强化学习（MARL）与进化博弈（如Fermi模仿规则）的结合机制。研究的重点在于如何通过引入不同形式的学习算法（Q-Learning, PPO）和记忆机制，促进群体合作行为的演化。

## 🌟 主要算法与创新点

项目中包含多种不同的策略演化与学习框架：

1. **Fermi Adaptive Q-Learning (基于自适应记忆的Fermi-Q学习)**
   - 核心代码：`2D.py`, `grid.py`, `diff-b.py`
   - 特点：将传统的Fermi模仿规则与Q-Learning相结合，并引入了自适应的记忆长度机制。智能体会根据历史收益动态调整其Q值更新策略。

2. **Deep Fusion Q-Learning (深度融合Q学习)**
   - 核心代码：`qp.py`, `wave_prediction.py`
   - 特点：利用累积收益不等式记忆方法，深度融合个体学习与社会模仿，探索合作倾向的演化。

3. **Meta-Imitation PPO (元模仿PPO)**
   - 核心代码：`snapshots/Meta imitation ppo.py`
   - 特点：采用PPO（近端策略优化）作为“元策略”来决定**“何时执行Fermi模仿”**。
     - **动作空间**：选择自主决策或执行Fermi规则模仿邻居。
     - **目的**：使智能体学会根据当前局势动态切换学习模式。

4. **Dual-Track PPO (双轨PPO网络)**
   - 核心代码：`snapshots_b/双轨ppo.py`
   - 特点：为合作者（C）和背叛者（D）分别建立两套PPO网络模型：
     - **CoopNet**：学习“何时维持合作”（基于反事实收益计算）。
     - **DefNet**：学习“何时转向合作”（基于合作网络带来的潜在收益提升）。

## 📂 项目结构

- **根目录文件**:
  - `2D.py`, `grid.py`: Fermi自适应Q学习的主要仿真脚本，包含网格环境构建与加速。
  - `qp.py`, `wave_prediction.py`: 深度融合Q学习机制实现。
  - `10.1.py`, `10.2.py` 等: 不同参数或演化规则的对比实验脚本。
- **`snapshots/` & `snapshots_b/`**:
  - 存放PPO相关的高级探索框架代码（如 `Meta imitation ppo.py`，`双轨ppo.py`），以及训练过程中生成的快照、实验结果数据 (`.npy`) 和对比图表 (`.png`)。
- **图表与视频输出 (`*.png`, `*.pdf`, `*.avi`)**: 
  - 实验结果的输出记录，包括合作率随时间/参数 $b$ 的演化图、相变图（Phase Transition）、空间斑图（Spatial Patterns）。

## 🛠️ 依赖与环境

本项目主要基于 Python 开发，性能敏感的大规模网格交互部分使用了 `Numba` 进行 JIT 加速。主要依赖包如下：

- `numpy`
- `matplotlib`
- `numba` (关键依赖：用于加速网格中数以万计智能体的交互运算)
- `torch` (PyTorch，用于 PPO 和深度强化学习相关算法)

**安装依赖**：
```bash
pip install numpy matplotlib numba torch
```

## 🚀 如何运行

1. **运行基于自适应Q学习的网格演化**:
   ```bash
   python grid.py
   ```
   *将会在终端输出演化进度，并自动保存结果图片（如相变图、合作率演化图）。*

2. **运行深度PPO架构网络实验**:
   ```bash
   python "snapshots_b/双轨ppo.py"
   ```

## 📊 结果可视化与分析

算法运行后生成的图表和数据可用于深入分析以下现象：
- **合作率演化 (Cooperation Evolution)**：不同背叛诱惑值（$b$ 参数）下，群体合作率随时间步的动态变化。
- **相变现象 (Phase Transition)**：在参数空间中（例如背叛诱惑值 $b$ 与记忆长度 $M$）合作者与背叛者的稳态比例变化。
- **空间分布斑图 (Spatial Patterns)**：可视化不同时间步下的合作者/背叛者空间分布，观察“合作簇 (Cooperation Clusters)”的形成、扩张或崩溃。

---
*Generated based on codebase analysis.*