import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from collections import deque
import numpy as np
import random
from Gridworld import Gridworld
import copy

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        if len(self.buffer) < self.sample_size:
            return iter([])
        batch = random.sample(self.buffer, self.sample_size)
        for exp in batch:
            yield exp

class LitDQN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        self.target_net = copy.deepcopy(self.net)
        
        self.buffer = deque(maxlen=1000)
        self.gamma = 0.9
        self.epsilon = 0.3
        self.batch_size = 200
        self.game = Gridworld(size=4, mode='random')
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.state = self.get_state(self.game)
        self.sync_rate = 500
        self.global_step_count = 0
        self.max_moves = 50
        self.moves = 0

        # Fill buffer initially
        self.populate_buffer(self.batch_size)

    def get_state(self, game):
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        return torch.from_numpy(state_).float().squeeze()

    def populate_buffer(self, steps):
        for _ in range(steps):
            self.play_step()

    def play_step(self):
        if random.random() < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            q_values = self.net(self.state.unsqueeze(0))
            action = torch.argmax(q_values).item()

        self.game.makeMove(self.action_set[action])
        self.moves += 1
        reward = self.game.reward()
        
        done = False
        if reward != -1 or self.moves >= self.max_moves:
            done = True

        next_state = self.get_state(self.game)
        
        exp = (self.state, action, reward, next_state, done)
        self.buffer.append(exp)
        
        if done:
            self.game = Gridworld(size=4, mode='random')
            self.state = self.get_state(self.game)
            self.moves = 0
        else:
            self.state = next_state
            
        return reward, done

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # Training Tip 1: AdamW for weight decay (regularization)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Training Tip 2: Learning Rate Scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        # Play a step in the environment
        self.play_step()
        self.global_step_count += 1
            
        states, actions, rewards, next_states, dones = batch
        
        q_values = self.net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q_value = rewards + self.gamma * (1 - dones.float()) * max_next_q

        loss = nn.MSELoss()(q_value, target_q_value)
        self.log('train_loss', loss, prog_bar=True)
        
        if self.global_step_count % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        return loss

    # Training Tip 3: Gradient Clipping is handled directly in PyTorch Lightning's Trainer
    # e.g., Trainer(gradient_clip_val=1.0)

if __name__ == "__main__":
    print("Training DQN with PyTorch Lightning on random mode...")
    model = LitDQN()
    
    # Gradient clipping enabled here via gradient_clip_val
    trainer = pl.Trainer(max_epochs=50, max_steps=5000, gradient_clip_val=1.0, log_every_n_steps=10)
    trainer.fit(model)
    print("Training complete.")
