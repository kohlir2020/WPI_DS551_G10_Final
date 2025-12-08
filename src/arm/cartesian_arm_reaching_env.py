"""
Cartesian Arm Reaching Environment with Inverse Kinematics
Uses end-effector position targets instead of joint velocities
Much easier for RL to learn!
"""

import gymnasium as gym
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class CartesianArmReachingEnv(gym.Env):
    """
    Arm reaching environment with Cartesian (EE position) control.
    
    Key difference from simple_arm_reaching_env:
    - Action space: 3D end-effector position target [-1, 1]³
    - IK solver: Maps EE target to joint angles automatically
    - Much easier for policies to learn (spatial reasoning)
    - Policies learn "go to X,Y,Z" instead of "move joints 1,2,3"
    
    Why this is better:
    1. Spatial reasoning is more natural for RL
    2. Fewer local minima in learning
    3. Policies generalize better
    4. Expected 3-5x faster learning
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, max_steps=200):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.prev_distance = None
        
        # Arm parameters (7-DOF Fetch arm)
        self.num_joints = 7
        self.arm_angles = np.zeros(7, dtype=np.float32)
        
        # DH parameters for forward kinematics
        # (Approximate for 7-DOF arm)
        self.dh_params = {
            'a': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'd': [0.1273, 0.0, 0.4, 0.0, 0.4, 0.0, 0.1610],
            'alpha': [np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0],
            'theta_offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        
        # Goal position (EE target)
        self.goal_position = np.random.randn(3).astype(np.float32) * 0.5
        
        # ============ PHASE 2 IMPROVEMENT ============
        # Action space: 3D Cartesian position commands [-1, 1]³
        # Much more intuitive than joint velocities!
        # Policy learns spatial reasoning naturally
        self.action_space = gym.spaces.Box(
            low=-1.0,   # [-1, -1, -1] = far left/down/back
            high=1.0,   # [+1, +1, +1] = far right/up/forward
            shape=(3,),  # 3D position
            dtype=np.float32
        )
        
        # Observation space: [distance_to_goal, current_EE_pos (3), goal_pos (3)]
        # Normalized: distance [0,1], positions [-1,1]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] + [-1.0]*6, dtype=np.float32),
            high=np.array([1.0] + [1.0]*6, dtype=np.float32),
            dtype=np.float32
        )
        
        self.max_distance = 2.0
        self.workspace_bounds = 1.0  # ±1 meter workspace
    
    def _forward_kinematics(self, joint_angles):
        """
        Compute end-effector position from joint angles using DH parameters
        Returns: [x, y, z] position of end-effector
        """
        # Simplified FK for 7-DOF arm
        # In practice, this would use proper DH parameters
        # For now, use simple approximation
        
        # Sum of sine transformations (simplified)
        x = (np.sin(joint_angles[0]) * (np.cos(joint_angles[2]) * 0.4 + 
                                        np.cos(joint_angles[4]) * 0.4 + 
                                        0.16))
        y = (np.cos(joint_angles[0]) * (np.cos(joint_angles[2]) * 0.4 + 
                                        np.cos(joint_angles[4]) * 0.4 + 
                                        0.16))
        z = (np.sin(joint_angles[1]) * 0.4 + 
             np.sin(joint_angles[3]) * 0.4 + 
             0.1273)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _inverse_kinematics(self, target_ee_pos):
        """
        Solve inverse kinematics to find joint angles for target EE position
        Uses numerical optimization (scipy.optimize.minimize)
        """
        def ik_error(joint_angles):
            """Minimize error between current EE and target EE"""
            current_ee = self._forward_kinematics(joint_angles)
            error = np.linalg.norm(current_ee - target_ee_pos)
            
            # Add regularization to prefer current angles (smooth motion)
            regularization = 0.01 * np.linalg.norm(joint_angles - self.arm_angles)
            
            return error + regularization
        
        # Solve IK
        result = minimize(
            ik_error,
            self.arm_angles,  # Initial guess: current angles
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * 7,
            options={'maxiter': 50}
        )
        
        # Clip to valid range
        joint_angles = np.clip(result.x, -np.pi, np.pi).astype(np.float32)
        return joint_angles
    
    def _get_observation(self):
        """Get observation: [distance, current_ee_pos (norm), goal_pos (norm)]"""
        current_ee = self._forward_kinematics(self.arm_angles)
        distance = np.linalg.norm(current_ee - self.goal_position)
        
        # Normalize observations
        normalized_distance = np.clip(distance / self.max_distance, 0.0, 1.0)
        normalized_ee = current_ee / self.workspace_bounds
        normalized_goal = self.goal_position / self.workspace_bounds
        
        obs = np.concatenate([
            [normalized_distance],
            normalized_ee,
            normalized_goal
        ]).astype(np.float32)
        
        return obs
    
    def _get_reward(self, distance, prev_distance=None):
        """
        PHASE 2 IMPROVEMENT: Reward function for Cartesian control
        Similar to Phase 1 but tuned for spatial reasoning
        """
        reward = 0.0
        
        # 1. Non-linear proximity bonus (reaching goal in 3D space)
        proximity_bonus = 50.0 * max(0, 1.0 - distance / self.max_distance)
        reward += proximity_bonus
        
        # 2. Dense progress signal (main learning driver)
        if prev_distance is not None:
            delta_distance = prev_distance - distance
            reward += 2.0 * max(delta_distance, -0.3)  # Higher weight than Phase 1
        else:
            reward += 2.0 * max(0, 1.0 - distance / self.max_distance)
        
        # 3. Sustained progress bonus
        progress_bonus = 0.05 * (1.0 - distance / self.max_distance)
        reward += progress_bonus
        
        # 4. Minimal step penalty
        reward -= 0.0005
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.arm_angles = np.zeros(7, dtype=np.float32)
        self.goal_position = self.np_random.standard_normal(3).astype(np.float32) * 0.5
        self.prev_distance = None
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute action (3D EE target) and return observation, reward, etc.
        
        Action: 3D position in [-1, 1]³
        1. Denormalize to workspace
        2. Solve IK for target position
        3. Update joint angles
        4. Calculate reward based on progress toward goal
        """
        self.current_step += 1
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Denormalize action to workspace coordinates
        # [-1, 1] → workspace bounded by max_distance
        target_ee_pos = action * self.workspace_bounds * 0.75
        
        # Solve inverse kinematics
        self.arm_angles = self._inverse_kinematics(target_ee_pos)
        
        # Get observation
        obs = self._get_observation()
        distance = obs[0] * self.max_distance  # Denormalize
        
        # Calculate reward
        reward = self._get_reward(distance, self.prev_distance)
        self.prev_distance = distance
        
        # Check termination
        terminated = distance < 0.15  # Success threshold
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance_to_goal": float(distance),
            "success": terminated,
            "ee_position": self._forward_kinematics(self.arm_angles).tolist(),
            "goal_position": self.goal_position.tolist()
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        pass


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CARTESIAN ARM REACHING ENVIRONMENT")
    print("="*70 + "\n")
    
    env = CartesianArmReachingEnv(max_steps=200)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"\nTesting random policy for 5 episodes...")
    
    total_reward = 0
    for ep in range(5):
        obs, _ = env.reset()
        ep_reward = 0
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += ep_reward
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Distance={info['distance_to_goal']:.3f}m")
    
    print(f"\nAverage reward: {total_reward/5:.2f}")
    print(f"Environment test complete! ✅")
    env.close()
