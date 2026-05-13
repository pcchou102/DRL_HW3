# HW3-1: DQN Understanding Report (Static Mode)

## 1. Basic DQN Implementation for Static Environment
In the static GridWorld environment, the positions of the Player, Goal, Pit, and Wall remain constant across all episodes. This makes it an easy environment for an agent to learn, as the optimal path to the goal is always the same.

**Key Components of the Naive DQN:**
*   **State Representation:** The state is a flattened tensor of the GridWorld board (64 dimensions), representing the positions of the objects. We also add a small amount of random noise to prevent the network from getting stuck.
*   **Q-Network:** We use a simple Multi-Layer Perceptron (MLP) with 3 linear layers and ReLU activations. The input is the 64-D state, and the output is a 4-D vector representing the Q-values for the 4 possible actions (Up, Down, Left, Right).
*   **Epsilon-Greedy Policy:** The agent balances exploration and exploitation. With probability `epsilon`, it takes a random action; otherwise, it takes the action with the highest Q-value. `epsilon` decays over time.
*   **Loss Function & Optimization:** We use Mean Squared Error (MSE) loss between the predicted Q-values and the target Q-values. The target Q-value is calculated as $Y = R + \gamma \max(Q(S'))$, where $R$ is the reward, $\gamma$ is the discount factor, and $\max(Q(S'))$ is the maximum predicted Q-value for the next state. We optimize the network using the Adam optimizer.

In naive DQN, the network updates its weights immediately after every step using the transition $(S, A, R, S')$. This can lead to instability because consecutive samples are highly correlated.

## 2. Experience Replay Buffer
To solve the instability and correlation issues of naive DQN, we introduce an **Experience Replay Buffer**.

**How it works:**
*   **Storage:** Instead of learning from a transition and discarding it, we store the transition tuple $(S, A, R, S', \text{done})$ in a replay buffer (implemented as a Python `deque` with a maximum length).
*   **Sampling:** During training, we randomly sample a "mini-batch" of transitions from the buffer.
*   **Learning:** We compute the loss and update the network weights using this random mini-batch rather than the single most recent transition.

**Why it improves learning:**
1.  **Breaks Correlation:** By randomly sampling from past experiences, we break the temporal correlation between consecutive steps, satisfying the independent and identically distributed (i.i.d.) assumption of gradient descent.
2.  **Data Efficiency:** Rare but important transitions (like hitting the goal or falling into a pit) are kept in the buffer and can be learned from multiple times.
3.  **Stability:** The target Q-values and the gradients become much more stable because the mini-batch represents a broader distribution of the agent's experiences.

*In our static mode tests, incorporating the Experience Replay Buffer drastically smoothed out the loss curve and allowed the agent to converge to the optimal policy faster and more reliably.*
