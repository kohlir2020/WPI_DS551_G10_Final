import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNavigationAgent(nn.Module):
    def __init__(self, observation_size, action_size):
        super(SimpleNavigationAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(4)  # Random action
        with torch.no_grad():
            q_values = self(state)
            return torch.argmax(q_values).item()

class DQNAgent:
    def __init__(self, observation_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = SimpleNavigationAgent(observation_size, action_size).to(self.device)
        self.target_net = SimpleNavigationAgent(observation_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = []
        self.batch_size = 32
        
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 10000:
            self.memory.pop(0)
            
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch]
        
        # Prepare batch for training
        state_batch = torch.stack([s for s, _, _, _ in batch]).to(self.device)
        action_batch = torch.tensor([a for _, a, _, _ in batch], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float).to(self.device)
        next_state_batch = torch.stack([s for _, _, _, s in batch]).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + 0.99 * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())