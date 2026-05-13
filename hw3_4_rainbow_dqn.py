"""
HW3-4（加分題）: Rainbow DQN — Random Mode（純 PyTorch 實作，修正版）
=======================================================================
整合以下 5 個 Rainbow 元件：
  1. Double DQN          — 解決 Q 值高估
  2. Dueling Network     — 分離 V(s) 與 A(s,a)
  3. Prioritized Experience Replay (PER) — SumTree 優先抽樣
  4. Multi-step Returns  — N 步累積回報（N=3）
  5. Noisy Linear Layers — 參數化噪聲取代 ε-greedy 探索

【修正記錄 v2】
  Bug1: N-step buffer 未正確處理中間步 done → 重寫 NStepBuffer 為滑窗式
  Bug2: NoisyNet 初期探索不足 → 前 1000 episode 加入 ε-greedy 暖機
  Bug3: IS 權重未正規化導致梯度爆炸 → 除以 max(w) 正規化
  Bug4: Double DQN 選動作時未重置噪聲 → 拆開 no_grad 區塊
  Bug5: GAMMA=0.9 對 random mode 路徑太短視 → 改為 0.99
  Bug6: Buffer=5000、epochs=3000 不足 → 改為 20000、5000
"""

import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from Gridworld import Gridworld


# ══════════════════════════════════════════════
# 1. Noisy Linear Layer
# ══════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """
    參數化噪聲線性層（Factorized Gaussian Noise）。
    w = μ_w + σ_w ⊙ ε_w，b = μ_b + σ_b ⊙ ε_b
    訓練時加噪聲（探索），eval() 時使用純均值（利用）。
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.sigma_init = sigma_init
        self._init_params()
        self.reset_noise()

    def _init_params(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return nn.functional.linear(x, w, b)


# ══════════════════════════════════════════════
# 2. Dueling + Noisy 網路架構
# ══════════════════════════════════════════════
class RainbowNet(nn.Module):
    """
    Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
    共享層使用普通 Linear，Value/Advantage 分支使用 NoisyLinear。
    """
    def __init__(self, input_dim: int = 64, hidden: int = 256, num_actions: int = 4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, 128), nn.ReLU(),
            NoisyLinear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, 128), nn.ReLU(),
            NoisyLinear(128, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ══════════════════════════════════════════════
# 3. SumTree
# ══════════════════════════════════════════════
class SumTree:
    """O(log n) 優先採樣樹"""
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = np.empty(capacity, dtype=object)
        self.n_entries = 0
        self.write_ptr = 0

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def add(self, priority: float, data):
        leaf_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def get(self, s: float):
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


# ══════════════════════════════════════════════
# 4. Prioritized Experience Replay
# ══════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """
    PER：TD Error 大的樣本優先被抽到。
    IS 權重修正因優先抽樣帶來的偏差。
    """
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta: float = 0.4, beta_inc: float = 5e-5):
        self.tree         = SumTree(capacity)
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta         = beta
        self.beta_inc     = beta_inc
        self.max_priority = 1.0

    def add(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int):
        indices, transitions, weights = [], [], []
        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_inc)

        # 取得所有葉節點中最小非零 priority，作為 IS 正規化基準
        leaves = self.tree.tree[-self.tree.capacity:]
        min_p  = float(np.min(leaves[leaves > 0])) / (self.tree.total_priority + 1e-8)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total_priority)
                leaf_idx, priority, data = self.tree.get(s)
            prob = (priority / (self.tree.total_priority + 1e-8))
            # 【修正】IS 權重正規化，避免數值爆炸
            w = (prob / (min_p + 1e-8)) ** (-self.beta)
            indices.append(leaf_idx)
            transitions.append(data)
            weights.append(w)

        # 再次正規化到 [0,1]
        weights = np.array(weights, dtype=np.float32)
        weights /= (weights.max() + 1e-8)
        return indices, transitions, weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            p = (abs(float(td)) + 1e-6) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


# ══════════════════════════════════════════════
# 5. N-step Buffer（修正版：滑窗 + 中間 done 截斷）
# ══════════════════════════════════════════════
class NStepBuffer:
    """
    【修正版】採用滑窗方式：每次 add() 後若 buffer 長度 >= N，
    就 get_transition() 取出最舊的 N-step transition，再 pop_left()。

    關鍵修正：計算 G 時，若中間某步 done=True，
    立刻截斷累積，並將 done_n 標為 True，
    防止已終止的遊戲繼續 bootstrap。
    """
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma  = gamma
        self.buffer = deque()

    def add(self, transition):
        self.buffer.append(transition)

    def get_transition(self):
        """取出以 buffer[0] 為起點的 N-step transition"""
        if len(self.buffer) < self.n_step:
            return None
        state0, action0 = self.buffer[0][0], self.buffer[0][1]
        G      = 0.0
        done_n = False
        last_s2 = self.buffer[self.n_step - 1][3]

        for k in range(self.n_step):
            _, _, r, s2, d = self.buffer[k]
            G += (self.gamma ** k) * r
            last_s2 = s2
            if d:
                done_n = True
                break   # 中間遇到終止，截斷不再往後累積
        return (state0, action0, G, last_s2, done_n)

    def pop_left(self):
        if self.buffer:
            self.buffer.popleft()

    def is_ready(self):
        return len(self.buffer) >= self.n_step

    def clear(self):
        self.buffer.clear()


# ══════════════════════════════════════════════
# 6. 主訓練函式（修正版）
# ══════════════════════════════════════════════
def train_rainbow(mode: str = 'random', epochs: int = 5000):
    """
    Rainbow DQN 訓練（全部 Bug 修正版）。

    主要改動：
      - GAMMA 0.9 → 0.99（random mode 需要更長遠的視野）
      - Buffer 5000 → 20000（增加樣本多樣性）
      - 前 1000 episodes ε-greedy 暖機（解決初期探索不足）
      - IS 權重正規化（防止梯度爆炸）
      - N-step 滑窗 + 中間 done 截斷
      - Double DQN 正確重置噪聲
      - 獎勵縮放 (/10)
      - Cosine Annealing 學習率排程
    """
    # ── 超參數 ──
    BATCH_SIZE  = 128
    GAMMA       = 0.99      # 提高折扣率
    N_STEP      = 3
    SYNC_FREQ   = 400
    MAX_MOVES   = 50
    LR          = 1e-3
    WARMUP_EPIS = 1000      # ε-greedy 暖機 episode 數
    EPS_START   = 1.0
    EPS_END     = 0.05

    # ── 建立網路 ──
    online_net = RainbowNet(input_dim=64, hidden=256, num_actions=4)
    target_net = RainbowNet(input_dim=64, hidden=256, num_actions=4)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(reduction='none')

    # ── Replay Buffer ──
    per_buffer = PrioritizedReplayBuffer(capacity=20000, alpha=0.6, beta=0.4)
    n_step_buf = NStepBuffer(n_step=N_STEP, gamma=GAMMA)

    action_set  = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses      = []
    win_history = []
    step_cnt    = 0

    def get_state(game):
        s = game.board.render_np().reshape(64).astype(np.float32)
        s += np.random.rand(64).astype(np.float32) / 100.0
        return torch.from_numpy(s)

    # ── 預先填充 Buffer（純隨機）──
    print("  預先填充 Replay Buffer...")
    pre_game, pre_moves = Gridworld(size=4, mode=mode), 0
    while len(per_buffer) < max(BATCH_SIZE * 4, 512):
        s  = get_state(pre_game)
        a  = np.random.randint(0, 4)
        pre_game.makeMove(action_set[a])
        pre_moves += 1
        raw_r = pre_game.reward()
        r     = raw_r / 10.0
        done  = (raw_r != -1) or (pre_moves >= MAX_MOVES)
        s2    = get_state(pre_game)
        n_step_buf.add((s, a, r, s2, done))
        if n_step_buf.is_ready():
            t = n_step_buf.get_transition()
            if t:
                per_buffer.add(t)
            n_step_buf.pop_left()
        if done:
            pre_game, pre_moves = Gridworld(size=4, mode=mode), 0
            n_step_buf.clear()
    print(f"  預填完成（{len(per_buffer)} 筆）")

    # ── 主訓練迴圈 ──
    for episode in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state  = get_state(game)
        n_step_buf.clear()
        mov, ep_won = 0, False

        # 計算當前 ε（線性衰減，暖機結束後固定 EPS_END）
        epsilon = EPS_END + max(0, (EPS_START - EPS_END) * (1 - episode / WARMUP_EPIS))

        while True:
            mov      += 1
            step_cnt += 1

            # ── 動作選擇 ──
            online_net.train()
            online_net.reset_noise()
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)      # ε-greedy 暖機
            else:
                with torch.no_grad():
                    action_ = int(torch.argmax(online_net(state.unsqueeze(0))).item())

            game.makeMove(action_set[action_])
            raw_r  = game.reward()
            reward = raw_r / 10.0
            done   = (raw_r != -1) or (mov >= MAX_MOVES)
            if raw_r == 10:
                ep_won = True
            state2 = get_state(game)

            # ── N-step 滑窗 ──
            n_step_buf.add((state, action_, reward, state2, done))
            if n_step_buf.is_ready():
                t = n_step_buf.get_transition()
                if t:
                    per_buffer.add(t)
                n_step_buf.pop_left()

            state = state2

            # ── 訓練更新 ──
            if len(per_buffer) >= BATCH_SIZE:
                idxs, batch, is_w = per_buffer.sample(BATCH_SIZE)

                s1_l, a_l, r_l, s2_l, d_l = zip(*batch)
                def to_tensor(lst, dtype):
                    return torch.stack([x.float() if isinstance(x, torch.Tensor)
                                        else torch.tensor(x, dtype=dtype) for x in lst])
                s1_b   = to_tensor(s1_l, torch.float)
                s2_b   = to_tensor(s2_l, torch.float)
                act_b  = torch.tensor(a_l, dtype=torch.long)
                rew_b  = torch.tensor(r_l, dtype=torch.float)
                done_b = torch.tensor(d_l, dtype=torch.float)
                w_b    = torch.tensor(is_w, dtype=torch.float)  # 已正規化

                online_net.train()

                # ── 先計算目標 Y（no_grad，不建計算圖）──
                # 必須在 Q_sa forward 之前完成 reset_noise，
                # 否則 inplace copy_() 會破壞 Q_sa 的計算圖。
                with torch.no_grad():
                    online_net.reset_noise()
                    best_a = torch.argmax(online_net(s2_b), dim=1)   # 主網路選動作
                    Q_next = target_net(s2_b).gather(1, best_a.unsqueeze(1)).squeeze(1)
                    Y      = rew_b + (GAMMA ** N_STEP) * (1.0 - done_b) * Q_next

                # ── 再計算當前 Q(s,a)（建計算圖，用於反傳）──
                online_net.reset_noise()   # 重置噪聲（此時計算圖尚未開始）
                Q_sa   = online_net(s1_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

                td_err = Q_sa - Y
                loss   = (w_b * loss_fn(Q_sa, Y)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                losses.append(loss.item())
                per_buffer.update_priorities(idxs, td_err.detach().numpy())

                if step_cnt % SYNC_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        scheduler.step()
        win_history.append(1 if ep_won else 0)

        if (episode + 1) % 500 == 0:
            wr  = np.mean(win_history[-500:]) * 100
            avg = np.mean(losses[-1000:]) if losses else float('nan')
            print(f"  Ep {episode+1:5d}/{epochs} | 近期勝率: {wr:5.1f}% "
                  f"| Loss: {avg:.4f} | ε: {epsilon:.3f} | Buf: {len(per_buffer)}")

    return online_net, losses, win_history


# ══════════════════════════════════════════════
# 7. 評估函式
# ══════════════════════════════════════════════
def test_rainbow(model, mode='random', max_games=500, max_moves=50):
    """貪婪策略評估勝率（eval 模式，NoisyNet 使用確定性均值）"""
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0
    model.eval()

    for _ in range(max_games):
        game = Gridworld(size=4, mode=mode)
        s_   = game.board.render_np().reshape(64).astype(np.float32) + np.random.rand(64).astype(np.float32) / 100.0
        s    = torch.from_numpy(s_)
        for _ in range(max_moves):
            with torch.no_grad():
                a = int(torch.argmax(model(s.unsqueeze(0))).item())
            game.makeMove(action_set[a])
            r = game.reward()
            if r == 10:
                wins += 1
                break
            elif r == -10:
                break
            s_ = game.board.render_np().reshape(64).astype(np.float32) + np.random.rand(64).astype(np.float32) / 100.0
            s  = torch.from_numpy(s_)

    return wins / max_games


# ══════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("HW3-4: Rainbow DQN（Random Mode，純 PyTorch，修正版 v2）")
    print("元件: Double DQN + Dueling + PER + N-step(3) + NoisyNet")
    print("=" * 60)

    EPOCHS = 5000
    model, losses, win_history = train_rainbow(mode='random', epochs=EPOCHS)

    win_rate = test_rainbow(model, mode='random')
    print(f"\n[結果] Rainbow DQN 最終勝率: {win_rate * 100:.1f}%")

    # ── 繪圖 ──
    def smooth(x, N=100):
        return np.convolve(x, np.ones(N) / N, mode='valid') if len(x) >= N else np.array(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"HW3-4: Rainbow DQN（Random Mode）  最終勝率: {win_rate*100:.1f}%",
                 fontsize=13, fontweight='bold')

    # Loss 曲線
    ax1.plot(losses, color='#3498db', alpha=0.25, linewidth=0.8, label='Loss（原始）')
    if len(losses) >= 100:
        ax1.plot(range(99, len(losses)), smooth(losses, 100),
                 color='#1a252f', linewidth=2, label='Loss（平滑）')
    ax1.set_title("訓練 Loss")
    ax1.set_xlabel("步數")
    ax1.set_ylabel("Weighted Huber Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 勝率曲線（每 50 episode 計算一次）
    win_smooth = smooth(win_history, 50) * 100
    ax2.plot(range(49, len(win_history)), win_smooth, color='#27ae60', linewidth=2)
    ax2.axhline(y=win_rate * 100, color='#e74c3c', linestyle='--',
                label=f'最終勝率 {win_rate*100:.1f}%')
    ax2.set_title("訓練過程勝率（50-episode 滑動平均）")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("勝率 (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw3_4_rainbow_loss.png")
    plt.savefig(save_path, dpi=150)
    print(f"[OK] 圖表已儲存至: {save_path}")

    print("\n" + "=" * 60)
    print("修正項目：")
    print("  [1] N-step buffer 中間 done 截斷（防止錯誤 bootstrap）")
    print("  [2] ε-greedy 暖機（前 1000 ep）+ NoisyNet 探索")
    print("  [3] IS 權重正規化（防梯度爆炸）")
    print("  [4] Double DQN 正確重置噪聲（兩次分開呼叫）")
    print("  [5] GAMMA 0.9 → 0.99（random mode 需要長遠視野）")
    print("  [6] Buffer 5000 → 20000，epochs 3000 → 5000")
    print("=" * 60)
