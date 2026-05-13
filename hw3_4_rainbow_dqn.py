"""
HW3-4（加分題）: Rainbow DQN — Random Mode（重製進階版）
=======================================================================
整合深度強化學習的 5 大先進核心技術（Rainbow 變體架構）：
  1. Double DQN          — 消除傳統 DQN 的過度樂觀 Q 值高估偏誤
  2. Dueling Network     — 分流狀態價值 V(s) 與動作優勢 A(s,a)，提升學習效率
  3. Prioritized Experience Replay (PER) — 基於 SumTree 實現高效 TD-error 優先抽樣
  4. Multi-step Returns  — 採用 N-step (N=3) 加速遠期獎勵回傳與收斂
  5. Noisy Linear Layers — 透過參數化自適應噪聲取代傳統 ε-greedy 進行深度探索

【優化與重製重點】
  - 數值穩定版 PER：採用標準文獻公式 w_i = (N * P(i))^-β / max(w)，徹底避免極端權重干擾。
  - 雙重探索機制：初期結合 ε-greedy 確保多樣化成功軌跡進入 Buffer，中後期依賴自適應噪聲網路。
  - 正確的隨機重置時序：嚴格確保 Double DQN 的目標計算與當前預測享有各自獨立乾淨的計算圖。
  - 提升特徵表達力：擴大隱藏層維度與特徵提取深度，以完美應對全隨機模式下高達 43,680 種棋盤排列。
"""

import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os
import matplotlib
import matplotlib.pyplot as plt

# 設定 Matplotlib 支援中文顯示與高畫質輸出
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from Gridworld import Gridworld


# ══════════════════════════════════════════════
# 1. Noisy Linear Layer (自適應探索噪聲層)
# ══════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear Layer (Fortunato et al., 2017)
    運用分解式高斯噪聲大幅減少隨機參數取樣開銷，賦予網路主動探索未知的強大能力。
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # 可訓練的平均值與標準差參數
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # 註冊不參與梯度的常數 Buffer 以存放取樣噪聲
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
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
        """重新對分解式高斯分佈進行隨機取樣"""
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# ══════════════════════════════════════════════
# 2. Dueling + Noisy 網路架構 (Rainbow 骨幹)
# ══════════════════════════════════════════════
class RainbowNet(nn.Module):
    """
    整合 Dueling 架構與 NoisyNet：
    透過強大的共享全連接層提取空間特徵，再分流計算 Value 與 Advantage。
    """
    def __init__(self, input_dim: int = 64, hidden: int = 256, num_actions: int = 4):
        super().__init__()
        # 增強版特徵提取器，強化隨機模式下的空間感知
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        # 狀態價值流 (Value Stream)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, 128, sigma_init=0.5),
            nn.ReLU(),
            NoisyLinear(128, 1, sigma_init=0.5)
        )

        # 動作優勢流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, 128, sigma_init=0.5),
            nn.ReLU(),
            NoisyLinear(128, num_actions, sigma_init=0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.feature_extractor(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # 中心化組合，確保優勢函數的可識別性
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def reset_noise(self):
        """遞迴重置所有子層的探索噪聲"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ══════════════════════════════════════════════
# 3. SumTree 資料結構 (高效優先權搜尋樹)
# ══════════════════════════════════════════════
class SumTree:
    """提供 O(log N) 高效抽樣與更新的二元樹結構"""
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
# 4. Prioritized Experience Replay (PER 緩衝區)
# ══════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """
    具備重要性採樣 (Importance Sampling) 權重校正的優先權回放緩衝區。
    採用極度穩定的標準除法正規化機制，徹底避免數值下溢或梯度爆炸。
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_inc: float = 2e-5):
        self.tree         = SumTree(capacity)
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta         = beta
        self.beta_inc     = beta_inc
        self.max_priority = 1.0

    def add(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int):
        indices, transitions, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(leaf_idx)
            transitions.append(data)
            priorities.append(priority)

        # 計算抽樣機率 P(i) = p_i / sum(p)
        sampling_probs = np.array(priorities, dtype=np.float32) / (self.tree.total_priority + 1e-8)
        
        # 標準 PER IS 權重計算公式：w_i = (N * P(i))^-β
        weights = (self.tree.n_entries * sampling_probs + 1e-8) ** (-self.beta)
        # 針對當前批次進行最大值正規化，保證權重落在 (0, 1] 區間
        weights /= (weights.max() + 1e-8)

        return indices, transitions, torch.from_numpy(weights).float()

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            p = (abs(float(td)) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


# ══════════════════════════════════════════════
# 5. N-step 累積回報器 (完美滑窗機制)
# ══════════════════════════════════════════════
class NStepBuffer:
    """
    精確計算多步折扣回報：G_t = r_t + γ*r_{t+1} + ... + γ^{N-1}*r_{t+N-1}
    若中途遭遇 Done 終止狀態，立即截斷後續 bootstrap，提供最純淨的監督訊號。
    """
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma  = gamma
        self.buffer = deque()

    def add(self, transition):
        self.buffer.append(transition)

    def get_transition(self):
        if len(self.buffer) < self.n_step:
            return None
        
        state0, action0 = self.buffer[0][0], self.buffer[0][1]
        discounted_reward = 0.0
        is_terminal       = False
        last_state        = self.buffer[-1][3]

        for k in range(self.n_step):
            _, _, r, s_next, done = self.buffer[k]
            discounted_reward += (self.gamma ** k) * r
            last_state = s_next
            if done:
                is_terminal = True
                break
                
        return (state0, action0, discounted_reward, last_state, is_terminal)

    def pop_left(self):
        if self.buffer:
            self.buffer.popleft()

    def is_ready(self):
        return len(self.buffer) >= self.n_step

    def clear(self):
        self.buffer.clear()


# ══════════════════════════════════════════════
# 6. 核心訓練引擎
# ══════════════════════════════════════════════
def train_rainbow(mode: str = 'random', epochs: int = 5000):
    """
    Rainbow DQN 進階重製版主體訓練流程。
    """
    # ── 嚴謹調校的超級參數 ──
    BATCH_SIZE   = 128
    GAMMA        = 0.99
    N_STEP       = 3
    SYNC_FREQ    = 500       # 目標網路硬同步頻率
    MAX_MOVES    = 50
    LR           = 5e-4      # 穩定適中的學習率搭配 LayerNorm
    BUFFER_CAP   = 20000
    WARMUP_EPIS  = 1200      # 延長初期混合探索期保證高質量軌跡收集

    # 建立雙重網路結構
    online_net = RainbowNet(input_dim=64, hidden=256, num_actions=4)
    target_net = RainbowNet(input_dim=64, hidden=256, num_actions=4)
    target_net.load_state_dict(online_net.state_dict())
    target_net.train()       # 確保目標網路同樣啟用隨機探索層進行分佈一致性取樣

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn   = nn.SmoothL1Loss(reduction='none')

    per_buffer = PrioritizedReplayBuffer(capacity=BUFFER_CAP, alpha=0.6, beta=0.4)
    n_step_buf = NStepBuffer(n_step=N_STEP, gamma=GAMMA)

    action_set  = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses      = []
    win_history = []
    global_step = 0

    def extract_state_tensor(game):
        """將棋盤轉為展平張量並注入極微量均勻擾動打破絕對對稱"""
        board_np = game.board.render_np().reshape(64).astype(np.float32)
        board_np += np.random.rand(64).astype(np.float32) / 100.0
        return torch.from_numpy(board_np)

    # ── 預先注入豐富多樣的隨機探索經驗 ──
    print("  [階段 1] 正在預先收集隨機軌跡填補緩衝區...")
    game_env = Gridworld(size=4, mode=mode)
    steps_played = 0
    while len(per_buffer) < 1000:
        s = extract_state_tensor(game_env)
        a = np.random.randint(0, 4)
        game_env.makeMove(action_set[a])
        steps_played += 1
        
        raw_reward = game_env.reward()
        reward_scaled = raw_reward / 10.0  # 獎勵縮放穩定梯度
        done = (raw_reward != -1) or (steps_played >= MAX_MOVES)
        s_next = extract_state_tensor(game_env)

        n_step_buf.add((s, a, reward_scaled, s_next, done))
        if n_step_buf.is_ready():
            transition = n_step_buf.get_transition()
            if transition:
                per_buffer.add(transition)
            n_step_buf.pop_left()

        if done:
            game_env = Gridworld(size=4, mode=mode)
            steps_played = 0
            n_step_buf.clear()
            
    print(f"  ✅ 緩衝區暖機完成，已就緒 {len(per_buffer)} 筆高質經驗樣本。")
    print(f"  [階段 2] 正式展開 Rainbow 深度網路自適應學習流程...")

    # ── 正式訓練迴圈 ──
    for episode in range(epochs):
        game_env = Gridworld(size=4, mode=mode)
        state    = extract_state_tensor(game_env)
        n_step_buf.clear()
        moves_cnt, is_episode_won = 0, False

        # 探索率調度：初期結合 ε-greedy 保證主動碰撞終點，後期交棒給 NoisyNet
        current_eps = max(0.02, 1.0 - (episode / WARMUP_EPIS))

        while True:
            moves_cnt   += 1
            global_step += 1

            online_net.train()
            online_net.reset_noise()

            # 混合探索決策機制
            if episode < WARMUP_EPIS and random.random() < current_eps:
                chosen_action = np.random.randint(0, 4)
            else:
                with torch.no_grad():
                    chosen_action = int(torch.argmax(online_net(state.unsqueeze(0))).item())

            game_env.makeMove(action_set[chosen_action])
            raw_reward = game_env.reward()
            reward_scaled = raw_reward / 10.0
            done = (raw_reward != -1) or (moves_cnt >= MAX_MOVES)
            
            if raw_reward == 10:
                is_episode_won = True
                
            next_state = extract_state_tensor(game_env)

            # N-step 滑動視窗儲存
            n_step_buf.add((state, chosen_action, reward_scaled, next_state, done))
            if n_step_buf.is_ready():
                transition = n_step_buf.get_transition()
                if transition:
                    per_buffer.add(transition)
                n_step_buf.pop_left()

            state = next_state

            # ── 執行神經網路深度優化更新 ──
            if len(per_buffer) >= BATCH_SIZE:
                indices, sample_batch, is_weights = per_buffer.sample(BATCH_SIZE)
                s0_batch, act_batch, rew_batch, s_last_batch, done_batch = zip(*sample_batch)

                s0_tensor     = torch.stack(s0_batch)
                s_last_tensor = torch.stack(s_last_batch)
                act_tensor    = torch.tensor(act_batch, dtype=torch.long)
                rew_tensor    = torch.tensor(rew_batch, dtype=torch.float)
                done_tensor   = torch.tensor(done_batch, dtype=torch.float)

                # 確保目標計算與當前預測享有完全獨立的隨機擾動狀態
                online_net.train()
                target_net.train()

                # [步驟 A] 計算 Double DQN 目標值 Y (禁用梯度追蹤)
                with torch.no_grad():
                    online_net.reset_noise()
                    target_net.reset_noise()
                    # 主網路選定最佳動作候選
                    optimal_actions = torch.argmax(online_net(s_last_tensor), dim=1)
                    # 目標網路評估該動作真實價值
                    next_q_values   = target_net(s_last_tensor).gather(1, optimal_actions.unsqueeze(1)).squeeze(1)
                    target_y        = rew_tensor + (GAMMA ** N_STEP) * (1.0 - done_tensor) * next_q_values

                # [步驟 B] 執行前向預測計算當前 Q(s, a)
                online_net.reset_noise()
                current_q_preds = online_net(s0_tensor).gather(1, act_tensor.unsqueeze(1)).squeeze(1)

                # 計算優先權誤差與加權損失
                td_errors     = current_q_preds - target_y
                weighted_loss = (is_weights * loss_fn(current_q_preds, target_y)).mean()

                optimizer.zero_grad()
                weighted_loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

                losses.append(weighted_loss.item())
                # 回傳絕對值更新 SumTree 優先權
                per_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

                # 定期硬同步權重至目標網路
                if global_step % SYNC_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        scheduler.step()
        win_history.append(1 if is_episode_won else 0)

        # 輸出高可讀性進度追蹤報告
        if (episode + 1) % 500 == 0:
            recent_win_rate = np.mean(win_history[-500:]) * 100.0
            avg_loss        = np.mean(losses[-1000:]) if losses else 0.0
            print(f"  ➜ 訓練進度: {episode+1:5d} / {epochs} 回合 │ 近期勝率: {recent_win_rate:5.1f}% │ "
                  f"平均損失: {avg_loss:.4f} │ 緩衝區容量: {len(per_buffer)}")

    return online_net, losses, win_history


# ══════════════════════════════════════════════
# 7. 模型泛化推論評估器
# ══════════════════════════════════════════════
def evaluate_rainbow_generalization(trained_model, mode='random', test_games=500, max_moves=50):
    """
    切換為 Eval 模式關閉內部隨機噪聲，純粹驗證神經網路學習到的空間泛化平均權重是否能穩健抵達終點。
    """
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    successful_runs = 0
    trained_model.eval()

    for _ in range(test_games):
        game_env = Gridworld(size=4, mode=mode)
        board_np = game_env.board.render_np().reshape(64).astype(np.float32) + np.random.rand(64).astype(np.float32) / 100.0
        state    = torch.from_numpy(board_np)

        for _ in range(max_moves):
            with torch.no_grad():
                q_predictions = trained_model(state.unsqueeze(0))
            best_action = int(torch.argmax(q_predictions).item())
            
            game_env.makeMove(action_set[best_action])
            reward = game_env.reward()
            
            if reward == 10:
                successful_runs += 1
                break
            elif reward == -10:
                break
                
            board_np = game_env.board.render_np().reshape(64).astype(np.float32) + np.random.rand(64).astype(np.float32) / 100.0
            state    = torch.from_numpy(board_np)

    return successful_runs / test_games


# ══════════════════════════════════════════════
# 主流程啟動區
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("HW3-4 終極優化版: Rainbow DQN 深度強化學習完整實作 (全隨機模式)")
    print("核心骨幹: Double DQN + Dueling Net + 數值穩定版 PER + N-step + NoisyNet")
    print("=" * 70)

    TOTAL_EPOCHS = 5000
    trained_model, training_losses, historical_wins = train_rainbow(mode='random', epochs=TOTAL_EPOCHS)

    print("\n  [階段 3] 正在進行嚴格的未見棋盤隨機排列推論泛化測試 (500 局獨立驗證)...")
    final_generalization_winrate = evaluate_rainbow_generalization(trained_model, mode='random')
    print(f"\n🏆 最終推論成功率 (Win Rate): {final_generalization_winrate * 100:.1f}%")

    # ── 繪製高視覺質感的平滑專業圖表 ──
    def compute_moving_average(data, window_size=100):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"HW3-4: Rainbow DQN 全隨機模式 (Random Mode) 最終推論勝率: {final_generalization_winrate*100:.1f}%",
                 fontsize=14, fontweight='bold', y=0.98)

    # 繪製損失收斂曲線
    axes[0].plot(training_losses, color='#2980b9', alpha=0.15, linewidth=0.5, label='即時批次損失')
    if len(training_losses) >= 100:
        axes[0].plot(range(99, len(training_losses)), compute_moving_average(training_losses, 100),
                     color='#2c3e50', linewidth=2.0, label='滑動平均損失 (n=100)')
    axes[0].set_title("神經網路加權損失函數收斂軌跡", fontsize=12)
    axes[0].set_xlabel("梯度更新次數 (Optimizer Steps)")
    axes[0].set_ylabel("Smooth L1 Loss ( Huber )")
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    # 繪製勝率攀升軌跡
    smoothed_wins = compute_moving_average(historical_wins, window_size=100) * 100.0
    axes[1].plot(range(99, len(historical_wins)), smoothed_wins, color='#27ae60', linewidth=2.2, label='訓練期勝率 (滑動視窗 n=100)')
    axes[1].axhline(y=final_generalization_winrate * 100.0, color='#e74c3c', linestyle='--', linewidth=1.8,
                    label=f'推論極限勝率 ({final_generalization_winrate*100:.1f}%)')
    axes[1].set_title("全隨機模式下 Agent 抵達終點成功率演進", fontsize=12)
    axes[1].set_xlabel("訓練回合 (Episodes)")
    axes[1].set_ylabel("成功抵達 Goal 比例 (%)")
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw3_4_rainbow_loss.png")
    plt.savefig(output_img_path, dpi=180)
    print(f"\n📊 訓練成果與學習軌跡圖表已成功渲染導出至: {output_img_path}")
    print("=" * 70)
