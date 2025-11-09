import habitat
import numpy as np
import torch

class HabitatEnv:
    def __init__(self, config_path):
        self.config = habitat.get_config(config_path)
        self.env = habitat.Env(config=self.config)
        
        # Define observation and action spaces
        self.observation_size = 128  # Will be adjusted based on actual observation space
        self.action_size = len(self.env.action_space)
        
    def reset(self):
        obs = self.env.reset()
        return self._process_observation(obs)
    
    def step(self, action):
        obs = self.env.step(action)
        return (
            self._process_observation(obs),
            obs["reward"],
            obs["done"],
            obs["info"]
        )
    
    def _process_observation(self, obs):
        """Convert Habitat observation to agent input format"""
        # This is a simplified version - modify based on your needs
        rgb = obs["rgb"]
        depth = obs["depth"]
        
        # Process the observation (example: flatten and normalize)
        processed_obs = np.concatenate([
            rgb.flatten() / 255.0,
            depth.flatten()
        ])
        
        return torch.FloatTensor(processed_obs)
    
    def close(self):
        self.env.close()