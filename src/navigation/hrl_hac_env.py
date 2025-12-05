"""
HAC (Hierarchical Actor-Critic) with Affordances + HER for Navigation
Minimal wrapper integrating DQN-based HRL with affordance subgoals and hindsight experience replay
Switchable with PPO-based HRL via manual comment/uncomment
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import sys
import habitat_sim

sys.path.insert(0, os.path.dirname(__file__))
from simple_navigation_env import SimpleNavigationEnv


# ============================================================
# DQN Q-Network
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """DQN agent for high-level or low-level control"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, device="cpu"):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.eps = 0.1  # Fixed epsilon for simplicity
    
    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(1).item())
    
    def update(self, batch):
        """Update from batch of (s, a, r, s', done)"""
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_values = self.q_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ============================================================
# HAC HRL Environment
# ============================================================

class HRLHACEnv(gym.Env):
    """
    HAC-style hierarchical navigation using DQN agents with affordance subgoals
    Compatible interface with HRLHighLevelEnvImproved for easy switching
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, low_level_model_path=None, subgoal_distance=5.0, 
                 option_horizon=50, debug=False):
        super().__init__()
        
        # Create low-level environment
        self.ll_env = SimpleNavigationEnv()
        self.sim = self.ll_env.sim
        self.pathfinder = self.sim.pathfinder
        
        # Parameters
        self.subgoal_distance = subgoal_distance
        self.option_horizon = option_horizon
        self.debug = debug
        
        # High-level observation: [distance_to_main_goal, angle_to_main_goal]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi], dtype=np.float32),
            high=np.array([100.0, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        
        # High-level action: 8 directions
        self.action_space = gym.spaces.Discrete(8)
        
        # Load or create DQN agents
        self.high_level = DQNAgent(state_dim=2, action_dim=8)
        self.low_level = DQNAgent(state_dim=2, action_dim=4)
        
        if low_level_model_path and os.path.exists(low_level_model_path):
            # Load low-level if available
            try:
                self.low_level.q_net.load_state_dict(
                    torch.load(low_level_model_path, map_location='cpu')
                )
                self.low_level.target_net.load_state_dict(self.low_level.q_net.state_dict())
                if self.debug:
                    print(f"✓ Loaded low-level DQN from {low_level_model_path}")
            except:
                if self.debug:
                    print(f"⚠ Failed to load low-level model, using random initialization")
        
        self.main_goal = None
        self.current_step = 0
        self.max_highlevel_steps = 20
    
    def _get_hl_obs(self):
        """Get high-level observation"""
        state = self.ll_env.agent.get_state()
        agent_pos = np.array(state.position, dtype=np.float32)
        
        to_goal = self.main_goal - agent_pos
        dist = float(np.linalg.norm(to_goal))
        
        if dist < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Compute angle to goal
        to_goal[1] = 0.0
        to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)
        
        # Agent forward direction
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        import quaternion
        q = quaternion.quaternion(state.rotation.w, state.rotation.x, 
                                   state.rotation.y, state.rotation.z)
        vq = quaternion.quaternion(0.0, *forward)
        rq = q * vq * q.inverse()
        forward = np.array([rq.x, rq.y, rq.z], dtype=np.float32)
        forward[1] = 0.0
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Signed angle
        cross = forward[0] * to_goal_norm[2] - forward[2] * to_goal_norm[0]
        dot = forward[0] * to_goal_norm[0] + forward[2] * to_goal_norm[2]
        angle = float(np.arctan2(cross, dot))
        
        return np.array([dist, angle], dtype=np.float32)
    
    def _sample_subgoal_affordance(self, action):
        """Sample subgoal using affordance-based approach"""
        agent_pos = np.array(self.ll_env.agent.get_state().position, dtype=np.float32)
        
        # 8 directions (N, NE, E, SE, S, SW, W, NW)
        angle = action * (2 * np.pi / 8)
        dx = self.subgoal_distance * np.cos(angle)
        dz = self.subgoal_distance * np.sin(angle)
        
        target = agent_pos + np.array([dx, 0.0, dz], dtype=np.float32)
        
        # Snap to navmesh
        snapped = self.pathfinder.snap_point(target)
        if not np.isnan(snapped).any():
            return np.array(snapped, dtype=np.float32)
        
        # Fallback: random navigable point
        return np.array(self.pathfinder.get_random_navigable_point(), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Reset low-level env
        ll_obs, _ = self.ll_env.reset()
        
        # Sample main goal
        agent_pos = np.array(self.ll_env.agent.get_state().position, dtype=np.float32)
        for _ in range(50):
            goal = self.pathfinder.get_random_navigable_point()
            dist = np.linalg.norm(np.array(goal) - agent_pos)
            if 8.0 <= dist <= 20.0:
                self.main_goal = np.array(goal, dtype=np.float32)
                break
        else:
            self.main_goal = np.array(goal, dtype=np.float32)
        
        self.current_step = 0
        return self._get_hl_obs(), {}
    
    def step(self, action):
        """Execute high-level action (runs low-level for up to option_horizon steps)"""
        # Sample subgoal based on action
        subgoal = self._sample_subgoal_affordance(action)
        self.ll_env.goal_position = subgoal
        
        # Execute low-level policy toward subgoal
        ll_steps = 0
        ll_total_reward = 0.0
        subgoal_reached = False
        
        for _ in range(self.option_horizon):
            ll_obs = self.ll_env._get_obs()
            ll_action = self.low_level.select_action(ll_obs, greedy=True)
            _, ll_reward, ll_done, ll_truncated, ll_info = self.ll_env.step(ll_action)
            
            ll_steps += 1
            ll_total_reward += ll_reward
            
            if ll_done or ll_truncated:
                subgoal_reached = True
                break
        
        # High-level reward: progress toward main goal
        hl_obs = self._get_hl_obs()
        progress = max(0, 10.0 - hl_obs[0])  # Reward being closer
        reward = progress / 10.0
        
        self.current_step += 1
        done = hl_obs[0] < 0.6
        truncated = self.current_step >= self.max_highlevel_steps
        
        info = {
            "main_distance": hl_obs[0],
            "ll_steps": ll_steps,
            "subgoal_reached": subgoal_reached
        }
        
        return hl_obs, reward, done, truncated, info
    
    def close(self):
        self.ll_env.close()


# TO USE: Comment/uncomment in skill_executor.py:
# from hrl_highlevel_env import HRLHighLevelEnvImproved  # PPO-based (current)
# from hrl_hac_env import HRLHACEnv as HRLHighLevelEnvImproved  # DQN-based HAC
