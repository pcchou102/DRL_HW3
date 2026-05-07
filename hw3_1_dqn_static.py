import numpy as np
import torch
import random
from matplotlib import pylab as plt
from Gridworld import Gridworld
from collections import deque
import copy

def train_dqn(mode='static', use_replay=True, epochs=1000):
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

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }
    
    losses = []
    
    if use_replay:
        mem_size = 1000
        batch_size = 200
        replay = deque(maxlen=mem_size)
        max_moves = 50
        
        for i in range(epochs):
            game = Gridworld(size=4, mode=mode)
            state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state1 = torch.from_numpy(state1_).float()
            status = 1
            mov = 0
            
            while(status == 1):
                mov += 1
                qval = model(state1)
                qval_ = qval.data.numpy()
                
                if (random.random() < epsilon):
                    action_ = np.random.randint(0,4)
                else:
                    action_ = np.argmax(qval_)
                
                action = action_set[action_]
                game.makeMove(action)
                
                state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
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
                        Q2 = model(state2_batch)
                    
                    Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
                    X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    
                    loss = loss_fn(X, Y.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                
                if reward != -1 or mov > max_moves:
                    status = 0
                    mov = 0
            if epsilon > 0.1:
                epsilon -= (1/epochs)
    else:
        for i in range(epochs):
            game = Gridworld(size=4, mode=mode)
            state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state = torch.from_numpy(state_).float()
            status = 1
            while(status == 1):
                qval = model(state)
                qval_ = qval.data.numpy()
                
                if (random.random() < epsilon):
                    action_ = np.random.randint(0,4)
                else:
                    action_ = np.argmax(qval_)
                
                action = action_set[action_]
                game.makeMove(action)
                
                state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
                state2 = torch.from_numpy(state2_).float()
                reward = game.reward()
                
                with torch.no_grad():
                    newQ = model(state2.reshape(1,64))
                maxQ = torch.max(newQ)
                
                if reward == -1:
                    Y = reward + (gamma * maxQ)
                else:
                    Y = reward
                
                Y = torch.Tensor([Y]).detach()
                X = qval.squeeze()[action_]
                
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                state = state2
                if reward != -1:
                    status = 0
            if epsilon > 0.1:
                epsilon -= (1/epochs)
                
    return model, losses

if __name__ == "__main__":
    print("Training Naive DQN on static mode...")
    model_naive, losses_naive = train_dqn(mode='static', use_replay=False, epochs=1000)
    print("Training DQN with Experience Replay on static mode...")
    model_replay, losses_replay = train_dqn(mode='static', use_replay=True, epochs=1000)
    print("Training complete.")
