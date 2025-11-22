# backup_simple_nav.py
import gymnasium as gym
import numpy as np

class GridWorldNav(gym.Env):
    """Ultra-simple gridworld for testing RL"""
    
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([size, size, size, size]),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size-1, self.size-1])
        return self._get_obs(), {}
    
    def step(self, action):
        # Move: 0=up, 1=down, 2=left, 3=right
        moves = np.array([[-1,0], [1,0], [0,-1], [0,1]])
        self.agent_pos = np.clip(self.agent_pos + moves[action], 0, self.size-1)
        
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if done else -0.01
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)