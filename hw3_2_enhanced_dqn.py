import numpy as np
import torch
import random
from Gridworld import Gridworld
from collections import deque
import copy

def train_double_dqn(mode='player', epochs=2000):
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500
    j = 0
    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch)
                
                with torch.no_grad():
                    # Double DQN logic: use main model to select action, target model to evaluate
                    Q_next_main = model(state2_batch)
                    best_actions = torch.argmax(Q_next_main, dim=1)
                    Q2 = model2(state2_batch)
                    Q_target = Q2.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                
                Y = reward_batch + gamma * ((1 - done_batch) * Q_target)
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0
    return model, losses

class DuelingDQN(torch.nn.Module):
    def __init__(self, l1=64, l2=150, l3=100, l4=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(l1, l2)
        self.relu1 = torch.nn.ReLU()
        
        # Value stream
        self.val_fc = torch.nn.Linear(l2, l3)
        self.val_out = torch.nn.Linear(l3, 1)
        
        # Advantage stream
        self.adv_fc = torch.nn.Linear(l2, l3)
        self.adv_out = torch.nn.Linear(l3, l4)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        
        val = torch.nn.ReLU()(self.val_fc(x))
        val = self.val_out(val)
        
        adv = torch.nn.ReLU()(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        return val + (adv - adv.mean(dim=1, keepdim=True))

def train_dueling_dqn(mode='player', epochs=2000):
    model = DuelingDQN()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500
    j = 0
    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0
    return model, losses

if __name__ == "__main__":
    print("Training Double DQN on player mode...")
    train_double_dqn(mode='player', epochs=100)
    print("Training Dueling DQN on player mode...")
    train_dueling_dqn(mode='player', epochs=100)
    print("Training complete.")
