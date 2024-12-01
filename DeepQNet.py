import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json

class DeepQNetwork(nn.Module):
    def __init__(self, map_shape=(40,40), players_state_size=6, action_size=4):
        super(DeepQNetwork, self).__init__()
        
        # Convolutional Layers for the game board
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Flatten output of convolutional layers
        conv_output_size = map_shape[0] * map_shape[1] * 64
        
        # Fully connected layers for the game board
        self.fc_player = nn.Linear(players_state_size, 128)
        
        # Combined layers
        self.fc_combined1 = nn.Linear(conv_output_size + 128, 128)
        self.fc_combined2 = nn.Linear(128, 128)
        self.fc_output = nn.Linear(128, action_size)
    
        
    def forward(self, map_input, players_state_input):
        # Convolutional pathway for the map
        x_map = F.relu(self.conv1(map_input))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        x_map = x_map.view(x_map.size(0), -1)  # Flatten
        
        # Dense pathway for the other state values
        x_player = F.relu(self.fc_player(players_state_input))
        
        # Combine the two pathways
        x = torch.cat((x_map, x_player), dim=1)
        x = F.relu(self.fc_combined1(x))
        x = F.relu(self.fc_combined2(x))
        x = self.fc_output(x)
        
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
        
        self.training_metrics = {
            "episode": [],
            "total_reward": [],
            "average_loss": [],
            "epsilon": []
        }
        
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
        map_input = torch.tensor(state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
        players_input = torch.tensor(state["player"], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(map_input, players_input).squeeze()
            
        
        
        for action in range(self.action_size):
            if action not in valid_actions:
                q_values[action] = -1e6
    
        # print(q_values)
                
        return torch.argmax(q_values).item()
        
        
    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return

    #     memBatch = random.sample(self.memory, batch_size)
        
    #     map_batch = []
    #     players_batch = []
    #     target_batch = []
        
    #     losses = []
        
    #     for state, action, reward, next_state, done in memBatch:
    #         map_input = torch.tensor(state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
    #         target = reward
    #         next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    #         with torch.no_grad():
    #             target += self.gamma * torch.max(self.model(next_state)[0]).item()
    
    #         current_q = self.model(state)
    #         target_f = current_q.clone().detach()
    #         target_f[0][action] = target
                        
    #         # print("current_q: ", current_q)
    #         # print("target_f: ", target_f)
            
    #         # Update Q-network
    #         self.optimizer.zero_grad()
    #         loss = self.criterion(current_q, target_f)
    #         loss.backward()
            
    #         # Apply gradient clipping
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            
    #         self.optimizer.step()
            
    #         losses.append(loss.item())
        
    #     avg_loss = sum(losses) / len(losses)
    #     print(f"Average Loss: {avg_loss}")
            
    #         # print("Loss: ", loss)
            
    #     # Decay epsilon - make exploration less likely over time
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a random batch of experiences
        memBatch = random.sample(self.memory, batch_size)
        
        # Prepare batched inputs
        map_batch = []
        players_batch = []
        targets = []
        
        
        for state, action, reward, next_state, done in memBatch:
            # Prepare current state inputs
            map_batch.append(torch.tensor(state["map"], dtype=torch.float32))  # Add channel dimension
            players_batch.append(torch.tensor(state["player"], dtype=torch.float32))
            
            # Compute the target Q-value
            target = reward
            if not done:
                next_map_input = torch.tensor(next_state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                next_players_input = torch.tensor(next_state["player"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    target += self.gamma * torch.max(self.model(next_map_input, next_players_input)).item()
            
            # Target tensor for the action taken
            targets.append((action, target))
        
        # Stack map and other inputs to form batched tensors
        map_batch = torch.stack(map_batch).unsqueeze(1)  # Shape: [batch_size, 1, 40, 40]
        players_batch = torch.stack(players_batch)  # Shape: [batch_size, 6]
        
        # Forward pass and compute loss for the batch
        current_q = self.model(map_batch, players_batch)
        target_q = current_q.clone().detach()

        for i, (action, target) in enumerate(targets):
            target_q[i][action] = target  # Update only the Q-value for the action taken
        
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        # print(f"Loss: {loss.item()}")
        
        
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
            
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    
    def train(self, env, episodes, batch_size, playerid):
        """Train the agent in the given environment."""
        
        for e in range(episodes):
            state = env.reset(model=self)
            
            map_input = torch.tensor(state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape [1, 1, 40, 40]
            players_input = torch.tensor(state["player"], dtype=torch.float32).unsqueeze(0)  # Shape [1, 6]
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            total_reward = 0
            done = False
            
            losses = []
            
            while not done:
                action = self.act(state, playerid)
                next_state, rewards, done = env.step()
                next_map_input = torch.tensor(next_state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                next_players_input = torch.tensor(next_state["player"], dtype=torch.float32).unsqueeze(0)
                # next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                # print(rewards)
                # print(env.players[1].playerid)
                reward = rewards[playerid]
                self.remember(
                    {"map": state["map"], "player": state["player"]},
                    action,
                    reward,
                    {"map": next_state["map"], "player": next_state["player"]},
                    done
                )
                map_input = next_map_input
                players_input = next_players_input
                total_reward += reward
                
                # Replay experience
                loss = self.replay(batch_size)
                if loss is not None:
                    losses.append(loss)
                    
            avg_loss = sum(losses) / len(losses) if losses else 0
            self.training_metrics["episode"].append(e)
            self.training_metrics["total_reward"].append(total_reward)
            self.training_metrics["average_loss"].append(avg_loss)
            self.training_metrics["epsilon"].append(self.epsilon)

            print(f"Episode {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
            
        self.save("deepq_model.pth")
        self.save_training_metrics()
        print("Training complete. Model saved to deepq_model.pth. Training metrics saved to training_metrics.json")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
        
    def save_training_metrics(self, filename="training_metrics.json"):
        with open(filename, 'w') as f:
            json.dump(self.training_metrics, f)
            print(f"Training metrics saved to {filename}")
        
    