"""
Simple Arm Reaching Environment (No Habitat dependency)
For testing GPU training pipeline without Habitat-sim
"""

import gymnasium as gym
import numpy as np


class SimpleArmReachingEnv(gym.Env):
    """
    Simplified arm reaching environment.
    - No Habitat dependency
    - 7-DOF arm reaching toward a goal
    - Continuous control
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, max_steps=200):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.prev_distance = None  # Track previous distance for reward shaping
        
        # Arm parameters
        self.num_joints = 7
        self.arm_angles = np.zeros(7, dtype=np.float32)
        self.goal_position = np.random.randn(3).astype(np.float32) * 0.5
        
        # Phase 1 Improvement #3: Reduced action bounds
        # Smaller action space = more stable learning
        # Cap velocity to 0.2 rad/s instead of 1.0 rad/s
        self.action_space = gym.spaces.Box(
            low=-0.2,  # rad/s (was -1.0)
            high=0.2,  # rad/s (was 1.0)
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Phase 1 Improvement #1: Normalized observation space
        # Normalize distance to [0, 1] for better learning
        # Normalize angles to [-1, 1] for consistency
        self.max_distance = 2.0  # Maximum reachable distance
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] + [-1.0]*7, dtype=np.float32),
            high=np.array([1.0] + [1.0]*7, dtype=np.float32),
            dtype=np.float32
        )
    
    def _get_observation(self):
        """Get current observation [distance, angles...]"""
        # Simulate forward kinematics (simplified)
        ee_pos = np.sum(np.sin(self.arm_angles[:3])) * np.array([1, 1, 1])
        distance = np.linalg.norm(ee_pos - self.goal_position)
        
        # Phase 1 Improvement #1: Normalize observations
        normalized_distance = np.clip(distance / self.max_distance, 0.0, 1.0)
        # Normalize angles from [-pi, pi] to [-1, 1]
        normalized_angles = self.arm_angles / np.pi
        
        obs = np.concatenate([[normalized_distance], normalized_angles])
        return obs.astype(np.float32)
    
    def _get_reward(self, distance, prev_distance=None):
        """
        Phase 1 Improvement #2: Better reward shaping
        Dense reward shaping with multiple components:
        1. Success bonus: Non-linear, increases as you get closer
        2. Distance delta: Dense progress signal
        3. Sustained progress bonus: Encourage consistent improvement
        4. Light step penalty: Encourage efficiency
        """
        reward = 0.0
        
        # 1. Non-linear success bonus based on proximity
        # As distance approaches 0, reward approaches 50
        # At distance=0.15: reward = 50 * (1 - 0.15/2.0) ≈ 46
        # At distance=0.3: reward = 50 * (1 - 0.3/2.0) = 42.5
        proximity_bonus = 50.0 * max(0, 1.0 - distance / self.max_distance)
        reward += proximity_bonus
        
        # 2. Dense reward: Change in distance (main learning signal)
        # Very important for stable learning
        if prev_distance is not None:
            delta_distance = prev_distance - distance  # positive = closer
            # Reward moving closer, lightly penalize moving away
            # 1 * moving 0.1m closer = +0.1 reward (dense signal)
            reward += 1.0 * max(delta_distance, -0.3)
        else:
            # First step bonus based on starting proximity
            reward += 1.0 * max(0, 1.0 - distance / self.max_distance)
        
        # 3. Sustained progress bonus
        # Small bonus for each step (encourages any progress)
        # Scaled by how close you are (closer = more bonus)
        progress_bonus = 0.05 * (1.0 - distance / self.max_distance)
        reward += progress_bonus
        
        # 4. Very light step penalty
        # Over 200 steps: -0.1 total penalty (almost negligible)
        reward -= 0.0005
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.arm_angles = np.zeros(7, dtype=np.float32)
        self.goal_position = self.np_random.standard_normal(3).astype(np.float32) * 0.5
        self.prev_distance = None  # Reset distance tracking
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return observation, reward, terminated, truncated, info"""
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update arm angles with action as velocities
        # Action is now smaller [-0.2, 0.2], so we can use 1.0 scaling
        self.arm_angles = np.clip(
            self.arm_angles + action * 1.0,
            -np.pi,
            np.pi
        ).astype(np.float32)
        
        # Get observation
        obs = self._get_observation()
        distance = obs[0]
        
        # Calculate reward with dense shaping (paper-based)
        reward = self._get_reward(distance, self.prev_distance)
        self.prev_distance = distance
        
        # Check termination
        terminated = distance < 0.15  # Success
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance_to_goal": float(distance),
            "success": terminated
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        pass
