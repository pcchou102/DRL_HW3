# 🎮 HW3: DQN 與進階變體實作

> **深度強化學習 Homework 3** — 使用 GridWorld 4×4 環境

## 📋 作業說明

| 題目 | 內容 | 環境模式 | 配分 |
|------|------|----------|------|
| HW3-1 | Naive DQN & Experience Replay | `static` | 30% |
| HW3-2 | Double DQN & Dueling DQN | `player` | 40% |
| HW3-3 | PyTorch Lightning DQN（含訓練技巧） | `random` | 30% |
| HW3-4 | Rainbow DQN（加分題） | `random` | 加分 |

### GridWorld 模式說明

| 模式 | Player | Goal / Pit / Wall | 難度 |
|------|--------|-------------------|------|
| `static` | 固定 (0,3) | 固定 | ⭐ 易 |
| `player` | 隨機 | 固定 | ⭐⭐ 中 |
| `random` | 隨機 | 全隨機 | ⭐⭐⭐ 難 |

---

## 📁 檔案結構

```
hw33/
├── Gridworld.py               # GridWorld 環境
├── GridBoard.py               # 棋盤底層實作
│
├── hw3_1_dqn_static.py        # HW3-1: Naive DQN vs Experience Replay
├── hw3_2_enhanced_dqn.py      # HW3-2: Double DQN vs Dueling DQN
├── hw3_3_lightning_dqn.py     # HW3-3: PyTorch Lightning DQN
├── hw3_4_rainbow_dqn.py       # HW3-4: Rainbow DQN（純 PyTorch）
│
├── HW3_1_Understanding_Report.md  # HW3-1 理解報告
├── HW3_4_Rainbow_DQN_Analysis.md  # HW3-4 分析報告
│
├── hw3_1_losses.png           # 訓練後自動產生
├── hw3_2_comparison.png       # 訓練後自動產生
├── hw3_3_loss.png             # 訓練後自動產生
└── hw3_4_rainbow_loss.png     # 訓練後自動產生
```

---

## ⚙️ 環境需求

```bash
pip install torch numpy matplotlib pytorch-lightning pandas
```

> Python 3.10+ 建議

---

## 🚀 執行方式

```bash
# 進入工作目錄
cd hw33

# HW3-1: Naive DQN vs Experience Replay（static mode）
python hw3_1_dqn_static.py

# HW3-2: Double DQN vs Dueling DQN（player mode）
python hw3_2_enhanced_dqn.py

# HW3-3: PyTorch Lightning DQN（random mode）
python hw3_3_lightning_dqn.py

# HW3-4: Rainbow DQN（random mode，加分題）
python hw3_4_rainbow_dqn.py
```

---

## 🧠 實作重點

### HW3-1：Naive DQN vs Experience Replay

**Naive DQN（無 Replay Buffer）**
- 每步立即用當前 transition $(s, a, r, s')$ 更新
- 問題：連續樣本高度相關 → 梯度不穩定

**Experience Replay Buffer**
- 將 transition 存入 `deque`，隨機抽取 mini-batch 訓練
- 破除時序相關性，大幅穩定 Loss 曲線

---

### HW3-2：Double DQN & Dueling DQN

**Double DQN** 解決 Q 值高估問題：
```
傳統 DQN:  Y = r + γ · max_a Q_target(s', a)
Double:    Y = r + γ · Q_target(s', argmax_a Q_main(s', a))
           主網路選動作，目標網路評估 → 避免樂觀偏誤
```

**Dueling DQN** 分離值函數：
```
Q(s, a) = V(s) + [A(s, a) - mean_a A(s, a)]
          ↑ 狀態價值   ↑ 相對優勢（中心化）
```

---

### HW3-3：PyTorch Lightning 訓練技巧

| 技巧 | 實作方式 | 效果 |
|------|---------|------|
| **AdamW + Weight Decay** | `weight_decay=1e-4` | 防止過擬合 |
| **StepLR 學習率排程** | `step_size=1000, gamma=0.9` | 訓練後期精細收斂 |
| **Gradient Clipping** | `Trainer(gradient_clip_val=1.0)` | 防止梯度爆炸 |

> ⚠️ Windows 用戶：已設定 `enable_checkpointing=False` 避免檔案鎖定錯誤

---

### HW3-4：Rainbow DQN（加分題）

整合 5 個 Rainbow 元件（純 PyTorch，無外部 RL 函式庫）：

| # | 元件 | 解決問題 | 實作類別 |
|---|------|---------|---------|
| 1 | **Double DQN** | Q 值高估偏差 | 訓練迴圈邏輯 |
| 2 | **Dueling Network** | 值函數學習效率 | `RainbowNet` |
| 3 | **PER (SumTree)** | 樣本效率低 | `SumTree` + `PrioritizedReplayBuffer` |
| 4 | **N-step Returns** (N=3) | 獎勵傳播慢 | `NStepBuffer` |
| 5 | **Noisy Linear** | ε-greedy 探索不足 | `NoisyLinear` |

**核心架構圖：**
```
Input(64) → Linear(128) → ReLU
              ├── NoisyLinear → ReLU → NoisyLinear(→1)   = V(s)
              └── NoisyLinear → ReLU → NoisyLinear(→4)   = A(s,a)
Q(s,a) = V(s) + [A(s,a) - mean(A)]
```

---

## 📊 預期結果

| 模型 | 模式 | 預期勝率 |
|------|------|---------|
| Naive DQN | static | ~85% |
| Replay DQN | static | ~95%+ |
| Double DQN | player | ~70% |
| Dueling DQN | player | ~75% |
| Lightning DQN | random | ~40% |
| Rainbow DQN | random | ~55%+ |

---

## 📚 參考資料

- [DRL in Action GitHub](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction)
- Mnih et al., 2015 — *Human-level control through deep reinforcement learning* (DQN)
- Van Hasselt et al., 2016 — *Deep Reinforcement Learning with Double Q-learning*
- Wang et al., 2016 — *Dueling Network Architectures for Deep Reinforcement Learning*
- Hessel et al., 2018 — *Rainbow: Combining Improvements in Deep Reinforcement Learning*
- Fortunato et al., 2017 — *Noisy Networks for Exploration*
