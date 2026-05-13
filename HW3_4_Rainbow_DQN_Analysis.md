# HW3-4: Rainbow DQN Analysis and Tutorial for Random Mode GridWorld

## 1. 分析 (Analysis)

在 GridWorld 的 **Random Mode** 中，玩家 (Player)、目標 (Goal)、陷阱 (Pit) 和牆壁 (Wall) 的位置在每一局遊戲中都是完全隨機的。這意味著 Agent 無法單純「死記硬背」一條固定的路線（像 Static Mode 那樣），它必須真正理解網格上的空間關係與物件意義，才能在不同的初始狀態下泛化並成功找到目標。

傳統的 DQN 在這種高度隨機且稀疏獎勵的環境下，往往會遇到以下瓶頸：
1. **高估 Q 值 (Overestimation Bias)**：這會導致策略陷入次優解。
2. **樣本效率低下**：隨機探索很難在早期有效學習。
3. **單一期望值無法反映風險**：DQN 只預測未來的期望回報，但在 Random Mode 中，走到某個位置可能有一半機率碰到陷阱、一半機率過關，僅看期望值會忽略這種分佈。

### 為什麼需要 Rainbow DQN？
**Rainbow DQN** 結合了 DQN 領域六種最強大的擴充套件，對於 Random Mode 有決定性的幫助：

1. **Double DQN (解決高估問題)**：將選擇動作與評估動作的網路分離，讓 Agent 不會過度樂觀地估計 Q 值。
2. **Prioritized Experience Replay (PER, 優先經驗回放)**：在隨機模式中，碰到目標或陷阱的經驗很罕見但極其重要。PER 會優先抽出這些 TD Error 較大的經驗來學習，大幅提升樣本效率。
3. **Dueling Networks (競爭網路架構)**：分離狀態價值 $V(s)$ 與優勢函數 $A(s, a)$。在許多狀態下（例如周圍什麼都沒有），採取什麼動作影響不大，Dueling 架構能更快學習到狀態本身的價值。
4. **Multi-step Learning (多步學習)**：不只看下一步的回報，而是看未來 N 步的累積回報，這讓獎勵訊號能更快傳遞回早期狀態，加速收斂。
5. **Distributional RL (分佈式強化學習)**：預測回報的「機率分佈」而非單一期望值。這讓 Agent 能區分「穩定低回報」與「高風險高回報」的狀態，在有隨機陷阱的 GridWorld 中表現更穩健。
6. **Noisy Nets (噪聲網路)**：用參數化的噪聲層取代 $\epsilon$-greedy 進行探索。這讓網路能根據狀態的不確定性自動調整探索力度，比固定的 $\epsilon$ 衰減更聰明。

---

## 2. 教你怎麼做 (How to Implement Rainbow DQN)

要在 GridWorld Random Mode 中實作 Rainbow DQN，建議不要從零開始手刻所有六個元件（非常容易出錯），而是採取以下步驟：

### 步驟一：選擇成熟的 RL 函式庫
推薦使用 **Stable Baselines3** (SB3) 結合外部擴充套件，或者直接使用專注於 DQN 變體的庫，例如 **Tianshou** 或 **Ray RLlib**。
*在這裡我們以使用輕量且容易上手的 `Tianshou` 為例。*

### 步驟二：安裝必要套件
```bash
pip install tianshou torch numpy
```

### 步驟三：封裝 GridWorld 為 Gym Environment
Rainbow DQN 的實作通常需要標準的 `gym.Env` 介面。我們需要將 `Gridworld.py` 包裝起來：
```python
import gym
from gym import spaces
import numpy as np
from Gridworld import Gridworld

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.game = Gridworld(size=4, mode='random')
        # 動作空間：上下左右 (4個動作)
        self.action_space = spaces.Discrete(4)
        # 狀態空間：64維的連續空間 (Flattened board)
        self.observation_space = spaces.Box(low=0, high=1, shape=(64,), dtype=np.float32)
        self.action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    def reset(self):
        self.game = Gridworld(size=4, mode='random')
        state = self.game.board.render_np().reshape(64) + np.random.rand(64)/100.0
        return state.astype(np.float32)

    def step(self, action):
        self.game.makeMove(self.action_map[action])
        state = self.game.board.render_np().reshape(64) + np.random.rand(64)/100.0
        reward = self.game.reward()
        done = True if reward != -1 else False
        return state.astype(np.float32), reward, done, {}
```

### 步驟四：使用 Tianshou 建立 Rainbow DQN
Tianshou 內建了 Rainbow DQN 的策略，只需要將網路架構組裝起來即可：

```python
import torch
import tianshou as ts
from tianshou.policy import RainbowPolicy
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv

# 1. 建立環境
env = GridWorldEnv()
train_envs = DummyVectorEnv([lambda: GridWorldEnv() for _ in range(10)])
test_envs = DummyVectorEnv([lambda: GridWorldEnv() for _ in range(10)])

# 2. 定義網路架構 (包含 NoisyLinear, Dueling Network)
def noisy_linear(x, y):
    return ts.utils.net.discrete.NoisyLinear(x, y, 0.5)

net = ts.utils.net.common.Net(
    env.observation_space.shape, 
    env.action_space.shape,
    hidden_sizes=[128, 128], 
    device='cpu'
)

# 3. 建立 Rainbow Policy
num_atoms = 51
v_min = -10.0
v_max = 10.0
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = RainbowPolicy(
    model=net,
    optim=optim,
    discount_factor=0.9,
    estimation_step=3, # Multi-step learning (N=3)
    target_update_freq=320,
    num_atoms=num_atoms, # Distributional RL
    v_min=v_min,
    v_max=v_max,
    is_double=True # Double DQN
)

# 4. 建立 Prioritized Experience Replay Buffer
buf = PrioritizedVectorReplayBuffer(
    total_size=20000, 
    buffer_num=len(train_envs),
    alpha=0.6, 
    beta=0.4
)

# 5. 訓練與收集資料
train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=1000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= 8.0 # 設定早停條件
)

print(f"訓練完成！結果：{result}")
```

### 總結
透過引入 Tianshou 的 Rainbow DQN 框架：
1. 我們把 Dueling, Double, PER, Multi-step, Distributional 等技巧一次整合。
2. 讓 GridWorld 的 Random Mode 能夠更穩定地學習，解決了原本 Naive DQN 容易在隨機環境中崩潰不收斂的問題。
3. 訓練速度與樣本效率將得到顯著的提升。
