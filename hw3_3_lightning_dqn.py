"""
HW3-3: DQN with PyTorch Lightning — Random Mode
=================================================
功能：
  將 PyTorch DQN 改寫為 PyTorch Lightning 框架，並整合三種訓練技巧：
  1. AdamW + 權重衰減（Weight Decay）：防止過擬合，正則化效果
  2. StepLR 學習率調度（LR Scheduling）：訓練後期降低學習率，精細收斂
  3. 梯度裁剪（Gradient Clipping）：透過 Trainer(gradient_clip_val=1.0) 實現

環境：GridWorld 4×4，mode='random'
  - Player/Goal/Pit/Wall 位置全部隨機 → 最難的泛化測試

Windows 注意事項：
  - 已停用 ModelCheckpoint（enable_checkpointing=False）
    → 避免 Windows 檔案鎖定（WinError 32）導致訓練中斷
  - Logger 使用 CSVLogger，輸出至 hw33_lightning_logs/ 資料夾
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers import CSVLogger
from collections import deque
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd
import copy
import os
from Gridworld import Gridworld


# ─────────────────────────────────────────────
# IterableDataset：從 Replay Buffer 隨機抽樣
# ─────────────────────────────────────────────
class RLDataset(IterableDataset):
    """
    繼承 IterableDataset，讓 PyTorch DataLoader 能從 Replay Buffer 抽樣。

    每個 Epoch 產生 steps_per_epoch 個 batch，
    每個 batch 從 buffer 中隨機取 batch_size 筆 transition。

    注意：必須 yield 個別樣本（非 batch），
    DataLoader 會自動將它們整理成 batch（collate）。
    """
    def __init__(self, buffer, steps_per_epoch=200, batch_size=200):
        self.buffer          = buffer
        self.steps_per_epoch = steps_per_epoch
        self.batch_size      = batch_size

    def __iter__(self):
        # buffer 尚未填滿時跳過
        if len(self.buffer) < self.batch_size:
            return

        # 每個 epoch 總共產生 steps_per_epoch * batch_size 筆樣本
        # DataLoader 每次取 batch_size 筆 → 共執行 steps_per_epoch 步
        for _ in range(self.steps_per_epoch * self.batch_size):
            yield random.choice(self.buffer)


# ─────────────────────────────────────────────
# LightningModule：DQN Agent
# ─────────────────────────────────────────────
class LitDQN(pl.LightningModule):
    """
    PyTorch Lightning 封裝的 DQN。
    Lightning 自動處理：訓練迴圈、裝置管理、日誌記錄、梯度裁剪等。
    """

    def __init__(self):
        super().__init__()

        # ── 主網路 & 目標網路 ──
        self.net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        self.target_net = copy.deepcopy(self.net)

        # ── 超參數 ──
        self.buffer      = deque(maxlen=2000)
        self.gamma       = 0.9
        self.epsilon     = 1.0    # 初始探索率（高探索）
        self.eps_min     = 0.05   # 最低探索率
        self.batch_size  = 200
        self.sync_rate   = 500    # 同步目標網路的頻率（步）
        self.max_moves   = 50     # 每局最多步數

        # ── 環境狀態 ──
        self.game  = Gridworld(size=4, mode='random')
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.state      = self._get_state()
        self.moves      = 0
        self._step_cnt  = 0   # 訓練步計數（用於同步目標網路 & ε 衰減）

        # ── 預先填充 buffer ──
        print("  預先填充 Replay Buffer...")
        self._populate_buffer(self.batch_size)
        print(f"  Buffer 已填充 {len(self.buffer)} 筆資料")

    # ── 取得當前環境狀態 ──
    def _get_state(self):
        state_ = self.game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        return torch.from_numpy(state_).float().squeeze()

    # ── 預先收集隨機經驗以填充 buffer ──
    def _populate_buffer(self, steps):
        for _ in range(steps):
            self._play_step(epsilon_override=1.0)  # 完全隨機探索

    # ── 在環境中執行一步 ──
    def _play_step(self, epsilon_override=None):
        eps = epsilon_override if epsilon_override is not None else self.epsilon

        if random.random() < eps:
            action = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                q_values = self.net(self.state.unsqueeze(0))
            action = torch.argmax(q_values).item()

        self.game.makeMove(self.action_set[action])
        self.moves += 1
        reward = self.game.reward()

        done = (reward != -1) or (self.moves >= self.max_moves)
        next_state = self._get_state()

        self.buffer.append((self.state, action, reward, next_state, done))

        if done:
            # 重置環境
            self.game  = Gridworld(size=4, mode='random')
            self.state = self._get_state()
            self.moves = 0
        else:
            self.state = next_state

        return reward, done

    # ── 前向傳播 ──
    def forward(self, x):
        return self.net(x)

    # ── 訓練單步（Lightning 自動呼叫） ──
    def training_step(self, batch, batch_idx):
        # 每個訓練步讓 agent 在環境中走一步
        self._play_step()
        self._step_cnt += 1

        # ── ε 衰減（每步線性衰減）──
        total_steps = 200 * 50  # steps_per_epoch * max_epochs
        self.epsilon = max(self.eps_min, 1.0 - (self._step_cnt / total_steps) * (1.0 - self.eps_min))

        # ── 解包 DataLoader 提供的 batch ──
        states, actions, rewards, next_states, dones = batch
        states      = states.float()
        actions     = actions.long()
        rewards     = rewards.float()
        next_states = next_states.float()
        dones       = dones.float()

        # ── 計算當前 Q 值 ──
        q_values  = self.net(states)
        q_value   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── 計算目標 Q 值（Target Network，不反傳梯度）──
        with torch.no_grad():
            next_q    = self.target_net(next_states)
            max_next  = torch.max(next_q, dim=1)[0]
            target_q  = rewards + self.gamma * (1 - dones) * max_next

        loss = nn.MSELoss()(q_value, target_q)
        self.log('train_loss', loss, prog_bar=True)
        self.log('epsilon',    self.epsilon, prog_bar=True)

        # ── 定期同步目標網路 ──
        if self._step_cnt % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss

    # ── 設定優化器與學習率排程器 ──
    def configure_optimizers(self):
        # 訓練技巧 1：AdamW（weight_decay 提供 L2 正則化效果）
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=1e-3,
            weight_decay=1e-4   # 防止過擬合
        )

        # 訓練技巧 2：StepLR 學習率排程
        #   每 1000 步將 lr 乘以 0.9 → 訓練後期更精細地更新
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.9
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ── DataLoader：從 Buffer 抽樣 ──
    def train_dataloader(self):
        dataset = RLDataset(self.buffer, steps_per_epoch=200, batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=0)


# ─────────────────────────────────────────────
# 訓練後繪圖
# ─────────────────────────────────────────────
def plot_training_results(log_dir):
    """
    從 CSVLogger 儲存的 metrics.csv 讀取並繪製 Loss 曲線。
    """
    csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"  找不到 metrics.csv：{csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'train_loss' not in df.columns:
        print("  metrics.csv 中無 train_loss 欄位")
        return

    df_loss = df.dropna(subset=['train_loss'])

    plt.figure(figsize=(10, 5))
    plt.plot(df_loss['step'], df_loss['train_loss'], color='#e67e22', alpha=0.4, linewidth=1, label='Loss（原始）')

    # 滑動平均平滑曲線
    if len(df_loss) >= 20:
        smooth = df_loss['train_loss'].rolling(window=20).mean()
        plt.plot(df_loss['step'], smooth, color='#c0392b', linewidth=2, label='Loss（平滑）')

    plt.title("HW3-3: PyTorch Lightning DQN Training Loss（Random Mode）", fontsize=13)
    plt.xlabel("訓練步數（Step）")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "hw3_3_loss.png")
    plt.savefig(save_path, dpi=150)
    print(f"  ✅ 訓練曲線已儲存至: {save_path}")


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("HW3-3: DQN with PyTorch Lightning（Random Mode）")
    print("訓練技巧: AdamW + StepLR + Gradient Clipping")
    print("=" * 55)

    # ── CSVLogger：輸出至 hw33 子資料夾（不寫到根目錄）──
    log_dir    = os.path.dirname(__file__)
    logger     = CSVLogger(save_dir=log_dir, name="hw3_3_lightning_logs")

    model = LitDQN()

    # 訓練技巧 3：Gradient Clipping（gradient_clip_val=1.0）
    # Lightning Trainer 自動在每次 optimizer.step() 前裁剪梯度
    trainer = pl.Trainer(
        max_epochs=50,
        max_steps=10000,
        gradient_clip_val=1.0,      # 梯度裁剪，防止梯度爆炸
        log_every_n_steps=10,
        enable_checkpointing=False,  # 停用 checkpoint（避免 Windows 檔案鎖定問題）
        logger=logger,
        enable_progress_bar=True,
    )

    print(f"\n開始訓練（max_steps=10000, gradient_clip_val=1.0）...")
    trainer.fit(model)

    print("\n訓練完成！")
    print(f"  最終 ε（探索率）: {model.epsilon:.4f}")

    # ── 讀取 log 並繪圖 ──
    log_version_dir = logger.log_dir
    print(f"  Log 儲存於: {log_version_dir}")
    plot_training_results(log_version_dir)

    print("\n" + "=" * 55)
    print("訓練技巧總結:")
    print("  [1] AdamW（weight_decay=1e-4）→ 防止過擬合")
    print("  [2] StepLR（step_size=1000, gamma=0.9）→ 學習率衰減")
    print("  [3] gradient_clip_val=1.0 → 防止梯度爆炸")
    print("=" * 55)
