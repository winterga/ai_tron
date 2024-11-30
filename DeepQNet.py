import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DeepQNetwork(nn.Module):
    def __init__(self, state_size=1606, action_size=4):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DeepQAgent:
    def __init__(self, gameObj, state_size, action_size, hidden_size, gamma, epsilon, epsilon_min, epsilon_decay, alpha=0.01):
        self.gameObj = gameObj
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000) # store maximum of 2000 states
        self.alpha = alpha
        
        self.model = DeepQNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        
    def act(self, state, playerid=2):
        valid_actions = [
            action for action in range(self.action_size) 
            if not self.gameObj.players[playerid].wouldCollide(action)
        ]
        
        if np.random.rand() <= self.epsilon: # explore; epsilon decides exploration threshold
            return random.choice(valid_actions) if valid_actions else random.randrange(self.action_size)
                
        # Exploitation
        return self.predict(state, valid_actions)
    
    def predict(self, state, valid_actions):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
            
        q_values = q_values.squeeze()
        print(len(q_values))
        print(self.action_size)
        for action in range(self.action_size):
            if action not in valid_actions:
                q_values[action] = -np.inf
                
        return torch.argmax(q_values).item()
        
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        memBatch = random.sample(self.memory, batch_size)
        losses = []
        
        for state, action, reward, next_state, done in memBatch:
            target = reward
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                target += self.gamma * torch.max(self.model(next_state)[0]).item()
    
            current_q = self.model(state)
            target_f = current_q.clone().detach()
            target_f[0][action] = target
                        
            # print("current_q: ", current_q)
            # print("target_f: ", target_f)
            
            # Update Q-network
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_f)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            
            self.optimizer.step()
            
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        print(f"Average Loss: {avg_loss}")
            
            # print("Loss: ", loss)
            
        # Decay epsilon - make exploration less likely over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    
    def train(self, env, episodes, batch_size, playerid):
        """Train the agent in the given environment."""
        
        for e in range(episodes):
            state = env.reset(model=self)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state, playerid)
                next_state, rewards, done = env.step()
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                # print(rewards)
                # print(env.players[1].playerid)
                reward = rewards[2]
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Replay experience
                self.replay(batch_size)

            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
            
        self.save("deepq_model.pth")
        print("Training complete. Model saved to deepq_model.pth")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
    