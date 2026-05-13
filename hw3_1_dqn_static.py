"""
HW3-1: Naive DQN — Static Mode
================================
功能：
  1. 實作不含 Experience Replay 的 Naive DQN（逐步更新）
  2. 實作含 Experience Replay Buffer 的 DQN（批次更新）
  3. 比較兩者的 Loss 曲線，並儲存圖表
  4. 測試訓練後的模型勝率

環境：GridWorld 4×4，mode='static'
  - Player 出現在 (0,3)，目標在 (0,0)，陷阱在 (0,1)，牆壁在 (1,1)
  - 所有物件位置固定，適合驗證基礎 DQN 邏輯
"""

import numpy as np
import torch
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
# 工具函式：測試已訓練模型的勝率
# ─────────────────────────────────────────────
def test_model(model, mode='static', max_games=200, max_moves=50):
    """
    讓模型以貪婪策略（epsilon=0）進行 max_games 局遊戲，回傳勝率。

    Args:
        model:      已訓練的 PyTorch 模型
        mode:       GridWorld 模式 ('static' / 'player' / 'random')
        max_games:  測試局數
        max_moves:  每局最多步數

    Returns:
        win_rate (float): 勝率 [0, 1]
    """
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0

    for _ in range(max_games):
        game = Gridworld(size=4, mode=mode)
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()

        for _ in range(max_moves):
            with torch.no_grad():
                qval = model(state)
            action_ = torch.argmax(qval).item()
            game.makeMove(action_set[action_])
            reward = game.reward()
            if reward == 10:          # 到達目標
                wins += 1
                break
            elif reward == -10:       # 掉入陷阱
                break
            state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state = torch.from_numpy(state_).float()

    return wins / max_games


# ─────────────────────────────────────────────
# HW3-1-A：Naive DQN（無 Experience Replay）
# ─────────────────────────────────────────────
def train_naive_dqn(mode='static', epochs=1000):
    """
    不使用 Experience Replay 的基礎 DQN。
    每一步立即用當前 transition (s, a, r, s') 更新網路權重。
    缺點：連續樣本高度相關 → 梯度更新不穩定。
    """
    # ── 網路架構：64 → 150 → 100 → 4（動作數）──
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 150),
        torch.nn.ReLU(),
        torch.nn.Linear(150, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 4)
    )

    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma     = 0.9
    epsilon   = 1.0          # 初始探索率

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        status = 1   # 1 = 遊戲進行中

        while status == 1:
            # ── ε-greedy 選擇動作 ──
            qval = model(state)
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)      # 探索
            else:
                action_ = torch.argmax(qval).item()    # 利用

            game.makeMove(action_set[action_])
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2  = torch.from_numpy(state2_).float()
            reward  = game.reward()

            # ── 計算 TD 目標 Q 值 ──
            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)

            # 若非終止狀態，加入折扣未來回報；否則直接用即時獎勵
            if reward == -1:
                Y = torch.tensor(reward + gamma * maxQ.item()).float()
            else:
                Y = torch.tensor(float(reward)).float()

            X = qval.squeeze()[action_]   # 預測 Q 值（scalar）

            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            state = state2
            if reward != -1:
                status = 0   # 遊戲結束（獲勝或失敗）

        # ε 線性衰減，最低保留 0.1
        if epsilon > 0.1:
            epsilon -= 1 / epochs

    return model, losses


# ─────────────────────────────────────────────
# HW3-1-B：DQN with Experience Replay Buffer
# ─────────────────────────────────────────────
def train_replay_dqn(mode='static', epochs=1000):
    """
    加入 Experience Replay Buffer 的 DQN。
    核心改進：
      1. 將每步 transition 存入 buffer（deque）
      2. 每步隨機抽出一個 mini-batch 進行批次更新
      3. 破除時間相關性 → 訓練更穩定、Loss 曲線更平滑
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 150),
        torch.nn.ReLU(),
        torch.nn.Linear(150, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 4)
    )

    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma     = 0.9
    epsilon   = 1.0

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []

    # ── Replay Buffer 設定 ──
    mem_size   = 1000   # buffer 最大容量
    batch_size = 200    # 每次抽樣的 mini-batch 大小
    replay     = deque(maxlen=mem_size)
    max_moves  = 50     # 每局最多步數（防止無限迴圈）

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state1  = torch.from_numpy(state1_).float()
        status  = 1
        mov     = 0

        while status == 1:
            mov += 1
            qval  = model(state1)
            qval_ = qval.data.numpy()

            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            game.makeMove(action_set[action_])
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2  = torch.from_numpy(state2_).float()
            reward  = game.reward()
            done    = (reward != -1)   # True 代表終止狀態

            # ── 存入 Replay Buffer ──
            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            # ── 當 buffer 足夠大時，進行批次訓練 ──
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)

                # 解包 mini-batch
                s1_b    = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                act_b   = torch.tensor([a  for (s1, a, r, s2, d) in minibatch], dtype=torch.long)
                rew_b   = torch.tensor([r  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)
                s2_b    = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_b  = torch.tensor([d  for (s1, a, r, s2, d) in minibatch], dtype=torch.float)

                Q1 = model(s1_b)
                with torch.no_grad():
                    Q2 = model(s2_b)

                # Bellman 方程式計算目標值（終止狀態不加未來回報）
                Y = rew_b + gamma * (1 - done_b) * torch.max(Q2, dim=1)[0]
                X = Q1.gather(dim=1, index=act_b.unsqueeze(1)).squeeze()

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            # 遊戲結束條件
            if reward != -1 or mov > max_moves:
                status = 0
                mov    = 0

        if epsilon > 0.1:
            epsilon -= 1 / epochs

    return model, losses


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("HW3-1: Naive DQN vs Experience Replay DQN")
    print("環境: GridWorld Static Mode (4×4)")
    print("=" * 50)

    EPOCHS = 1000

    # ── 訓練 Naive DQN ──
    print(f"\n[1/2] 訓練 Naive DQN（epochs={EPOCHS}）...")
    model_naive, losses_naive = train_naive_dqn(mode='static', epochs=EPOCHS)
    wr_naive = test_model(model_naive, mode='static')
    print(f"  → Naive DQN 勝率: {wr_naive*100:.1f}%")

    # ── 訓練 Replay DQN ──
    print(f"\n[2/2] 訓練 DQN with Experience Replay（epochs={EPOCHS}）...")
    model_replay, losses_replay = train_replay_dqn(mode='static', epochs=EPOCHS)
    wr_replay = test_model(model_replay, mode='static')
    print(f"  → Replay DQN 勝率: {wr_replay*100:.1f}%")

    # ── 繪製比較圖 ──
    def running_mean(x, N=30):
        """計算滑動平均，用於平滑 loss 曲線。"""
        if len(x) < N:
            return np.array(x)
        return np.convolve(x, np.ones(N) / N, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HW3-1: Naive DQN vs Experience Replay DQN (Static Mode)", fontsize=14, fontweight='bold')

    # 左圖：Naive DQN
    axes[0].plot(running_mean(losses_naive, 30), color='#e74c3c', label='Loss (平滑)')
    axes[0].set_title(f"Naive DQN\n勝率: {wr_naive*100:.1f}%")
    axes[0].set_xlabel("訓練步數")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右圖：Replay DQN
    axes[1].plot(running_mean(losses_replay, 30), color='#2ecc71', label='Loss (平滑)')
    axes[1].set_title(f"DQN + Experience Replay\n勝率: {wr_replay*100:.1f}%")
    axes[1].set_xlabel("訓練步數")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "hw3_1_losses.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] 訓練曲線已儲存至: {save_path}")

    print("\n" + "=" * 50)
    print("結果摘要:")
    print(f"  Naive DQN       → 勝率 {wr_naive*100:.1f}%")
    print(f"  Replay DQN      → 勝率 {wr_replay*100:.1f}%")
    print("=" * 50)
