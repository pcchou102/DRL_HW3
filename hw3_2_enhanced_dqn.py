"""
HW3-2: Enhanced DQN Variants — Player Mode
============================================
功能：
  1. 實作 Double DQN（解決 Q 值高估問題）
  2. 實作 Dueling DQN（分離 State Value 與 Advantage）
  3. 比較兩者訓練 Loss 曲線與最終勝率

環境：GridWorld 4×4，mode='player'
  - 只有 Player 位置隨機，其餘（Goal/Pit/Wall）固定
  - 比 static 難，需要策略泛化
"""

import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from Gridworld import Gridworld
from collections import deque
import copy
import os


# ─────────────────────────────────────────────
# 工具函式：測試模型勝率
# ─────────────────────────────────────────────
def test_model(model, mode='player', max_games=300, max_moves=50):
    """
    以貪婪策略評估模型，回傳勝率。

    Args:
        model:      已訓練模型（Sequential 或自定義 Module）
        mode:       GridWorld 模式
        max_games:  測試總局數
        max_moves:  每局最多步數

    Returns:
        win_rate (float)
    """
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0

    for _ in range(max_games):
        game = Gridworld(size=4, mode=mode)
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state  = torch.from_numpy(state_).float()

        for _ in range(max_moves):
            with torch.no_grad():
                qval = model(state)
            action_ = torch.argmax(qval).item()
            game.makeMove(action_set[action_])
            reward = game.reward()
            if reward == 10:
                wins += 1
                break
            elif reward == -10:
                break
            state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state  = torch.from_numpy(state_).float()

    return wins / max_games


# ─────────────────────────────────────────────
# HW3-2-A：Double DQN
# ─────────────────────────────────────────────
def train_double_dqn(mode='player', epochs=2000):
    """
    Double DQN 解決的問題：
      - 傳統 DQN 用同一網路「選動作」與「評估動作」
        → 容易系統性高估 Q 值 → 策略偏差

    解決方法：
      - 主網路 (model)  → 負責「選動作」（argmax）
      - 目標網路 (model2) → 負責「評估」所選動作的 Q 值
      - 每隔 sync_freq 步將主網路權重同步到目標網路
    """
    # ── 主網路與目標網路（起始時相同） ──
    model  = nn.Sequential(
        nn.Linear(64, 150), nn.ReLU(),
        nn.Linear(150, 100), nn.ReLU(),
        nn.Linear(100, 4)
    )
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma     = 0.9
    epsilon   = 1.0          # 從高探索率開始
    eps_min   = 0.05         # 最低探索率
    eps_decay = (1.0 - eps_min) / epochs  # 線性衰減

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    mem_size   = 1000
    batch_size = 200
    replay     = deque(maxlen=mem_size)
    max_moves  = 50
    sync_freq  = 500   # 每 500 步同步一次目標網路
    j          = 0     # 全域步數計數器
    losses     = []

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1  = torch.from_numpy(state1_).float()
        status  = 1
        mov     = 0

        while status == 1:
            j   += 1
            mov += 1
            qval  = model(state1)
            qval_ = qval.data.numpy()

            # ε-greedy 探索
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            game.makeMove(action_set[action_])
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2  = torch.from_numpy(state2_).float()
            reward  = game.reward()
            done    = (reward != -1)

            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)

                s1_b   = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                act_b  = torch.tensor([a  for (s1, a, r, s2, d) in minibatch], dtype=torch.long)
                rew_b  = torch.tensor([r  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)
                s2_b   = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_b = torch.tensor([d  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)

                Q1 = model(s1_b)

                with torch.no_grad():
                    # ── Double DQN 關鍵：主網路選動作，目標網路評估 ──
                    next_actions = torch.argmax(model(s2_b), dim=1)   # 主網路選最佳動作
                    Q2_target    = model2(s2_b)                        # 目標網路評估
                    Q_target     = Q2_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

                Y = rew_b + gamma * (1 - done_b) * Q_target
                X = Q1.gather(dim=1, index=act_b.unsqueeze(1)).squeeze()

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                # 定期將主網路權重複製到目標網路
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0
                mov    = 0

        # ε 線性衰減
        epsilon = max(eps_min, epsilon - eps_decay)

    return model, losses


# ─────────────────────────────────────────────
# HW3-2-B：Dueling DQN 網路架構
# ─────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling DQN 架構：
      Q(s, a) = V(s) + [A(s, a) - mean(A(s, ·))]

    分離的兩條「流」：
      - Value Stream：估計狀態本身的價值 V(s)
      - Advantage Stream：估計各動作相對優勢 A(s, a)

    好處：在許多狀態下，採取什麼動作影響不大。
    Dueling 架構讓模型能優先學習好的狀態，加速收斂。
    """
    def __init__(self, input_dim=64, hidden=150, advantage_hidden=100, num_actions=4):
        super(DuelingDQN, self).__init__()

        # 共享特徵提取層
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU()
        )

        # Value stream（輸出純量 V(s)）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, 1)
        )

        # Advantage stream（輸出向量 A(s, a)）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, num_actions)
        )

    def forward(self, x):
        features  = self.feature(x)
        value     = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)          # (batch, num_actions)

        # 合併：Q = V + (A - mean(A))
        # 減去均值確保 A 的唯一性（identifiability）
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


def train_dueling_dqn(mode='player', epochs=2000):
    """
    使用 Dueling DQN 架構訓練 Agent（搭配 Target Network）。
    訓練迴圈邏輯與 Double DQN 相同，差異僅在網路架構。
    """
    model  = DuelingDQN()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma     = 0.9
    epsilon   = 1.0
    eps_min   = 0.05
    eps_decay = (1.0 - eps_min) / epochs

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    mem_size   = 1000
    batch_size = 200
    replay     = deque(maxlen=mem_size)
    max_moves  = 50
    sync_freq  = 500
    j          = 0
    losses     = []

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1  = torch.from_numpy(state1_).float()
        status  = 1
        mov     = 0

        while status == 1:
            j   += 1
            mov += 1
            qval  = model(state1)
            qval_ = qval.data.numpy()

            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            game.makeMove(action_set[action_])
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2  = torch.from_numpy(state2_).float()
            reward  = game.reward()
            done    = (reward != -1)

            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)

                s1_b   = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                act_b  = torch.tensor([a  for (s1, a, r, s2, d) in minibatch], dtype=torch.long)
                rew_b  = torch.tensor([r  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)
                s2_b   = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_b = torch.tensor([d  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)

                Q1 = model(s1_b)
                with torch.no_grad():
                    Q2 = model2(s2_b)

                # 使用 Target Network 的 max Q 值
                Y = rew_b + gamma * (1 - done_b) * torch.max(Q2, dim=1)[0]
                X = Q1.gather(dim=1, index=act_b.unsqueeze(1)).squeeze()

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0
                mov    = 0

        epsilon = max(eps_min, epsilon - eps_decay)

    return model, losses


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("HW3-2: Double DQN vs Dueling DQN（Player Mode）")
    print("=" * 55)

    EPOCHS = 2000
    MODE   = 'player'

    # ── 訓練 Double DQN ──
    print(f"\n[1/2] 訓練 Double DQN（epochs={EPOCHS}）...")
    model_double, losses_double = train_double_dqn(mode=MODE, epochs=EPOCHS)
    wr_double = test_model(model_double, mode=MODE)
    print(f"  → Double DQN 勝率: {wr_double*100:.1f}%")

    # ── 訓練 Dueling DQN ──
    print(f"\n[2/2] 訓練 Dueling DQN（epochs={EPOCHS}）...")
    model_dueling, losses_dueling = train_dueling_dqn(mode=MODE, epochs=EPOCHS)
    wr_dueling = test_model(model_dueling, mode=MODE)
    print(f"  → Dueling DQN 勝率: {wr_dueling*100:.1f}%")

    # ── 繪製比較圖 ──
    def running_mean(x, N=50):
        if len(x) < N:
            return np.array(x)
        return np.convolve(x, np.ones(N) / N, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HW3-2: Double DQN vs Dueling DQN (Player Mode)", fontsize=14, fontweight='bold')

    # Double DQN Loss
    axes[0].plot(running_mean(losses_double, 50), color='#3498db', linewidth=1.5, label='Loss (平滑)')
    axes[0].set_title(f"Double DQN\n勝率: {wr_double*100:.1f}%")
    axes[0].set_xlabel("訓練步數")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Dueling DQN Loss
    axes[1].plot(running_mean(losses_dueling, 50), color='#9b59b6', linewidth=1.5, label='Loss (平滑)')
    axes[1].set_title(f"Dueling DQN\n勝率: {wr_dueling*100:.1f}%")
    axes[1].set_xlabel("訓練步數")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "hw3_2_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] 比較圖已儲存至: {save_path}")

    print("\n" + "=" * 55)
    print("結果比較:")
    print(f"  Double DQN  → 勝率 {wr_double*100:.1f}%  （解決 Q 值高估）")
    print(f"  Dueling DQN → 勝率 {wr_dueling*100:.1f}%  （分離 V/A 架構）")
    print("=" * 55)
