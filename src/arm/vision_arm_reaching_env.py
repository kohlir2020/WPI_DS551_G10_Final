"""
Vision-Based Arm Reaching Environment
Uses stacked camera observations from multiple viewpoints
Input: RGB image stacks instead of low-level state [distance, arm_angles]
"""

import gymnasium as gym
import numpy as np
import habitat_sim
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))


class VisionArmReachingEnv(gym.Env):
    """
    Vision-based arm reaching environment.
    
    Observation: Stacked RGB images from multiple cameras
    - Front camera (RGB)
    - Side camera (RGB)  
    - Top-down camera (RGB)
    - Depth-like sensor (optional)
    
    Stack 4 frames for temporal information (like Atari)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self,
                 scene_path=None,
                 goal_height=0.5,
                 success_distance=0.15,
                 max_steps=200,
                 frame_stack=4,
                 image_size=64):
        """
        Args:
            scene_path: Path to scene file
            goal_height: Height of goal (drawer/fridge handle)
            success_distance: Distance threshold for gripper trigger
            max_steps: Maximum steps per episode
            frame_stack: Number of frames to stack (temporal info)
            image_size: Resolution of images (image_size x image_size)
        """
        super().__init__()
        
        if scene_path is None:
            scene_path = os.path.join(
                os.path.dirname(__file__),
                "../../data/scene_datasets/habitat-test-scenes/apartment_1.glb"
            )
        
        scene_path = os.path.abspath(scene_path)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        
        print(f"Using scene: {scene_path}")
        
        # Initialize simulator
        self.sim = self._create_sim(scene_path, image_size)
        print("✓ Simulator initialized with vision sensors")
        
        # Parameters
        self.goal_height = goal_height
        self.success_distance = success_distance
        self.max_steps = max_steps
        self.current_step = 0
        self.goal_position = None
        
        # Vision parameters
        self.frame_stack = frame_stack
        self.image_size = image_size
        self.num_cameras = 3  # Front, side, top-down
        self.channels_per_camera = 3  # RGB
        
        # Frame stack for temporal information (like Atari)
        self.frame_buffers = [
            deque(maxlen=frame_stack) for _ in range(self.num_cameras)
        ]
        
        # Arm parameters (same as before)
        self.num_arm_joints = 7
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint"
        ]
        self.arm_angles = np.zeros(7, dtype=np.float32)
        
        # Action space: continuous velocities for 7 arm joints
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_arm_joints,),
            dtype=np.float32
        )
        
        # Observation space: Stacked RGB images from multiple cameras
        # Shape: (frame_stack, num_cameras, image_size, image_size, channels)
        # After concatenation: (frame_stack * num_cameras * channels, image_size, image_size)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                frame_stack * self.num_cameras * self.channels_per_camera,
                image_size,
                image_size
            ),
            dtype=np.uint8
        )
        
        print(f"✓ Vision-based arm environment initialized")
        print(f"  - Cameras: {self.num_cameras} (front, side, top-down)")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Frame stack: {frame_stack}")
        print(f"  - Observation shape: {self.observation_space.shape}")
        print(f"  - Goal height: {goal_height}m")
        print(f"  - Success distance: {success_distance}m")
        print(f"  - Max steps: {max_steps}")
    
    def _create_sim(self, scene_path, image_size):
        """Create simulator with vision sensors"""
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = 0  # Use GPU 0
        
        # Create agent with RGB sensors
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = []
        
        # Front RGB camera
        front_rgb = habitat_sim.CameraSensorSpec()
        front_rgb.uuid = "front_rgb"
        front_rgb.sensor_type = habitat_sim.SensorType.COLOR
        front_rgb.resolution = [image_size, image_size]
        front_rgb.position = [0.0, 0.6, 0.0]
        agent_cfg.sensor_specifications.append(front_rgb)
        
        # Side RGB camera (rotated 90 degrees)
        side_rgb = habitat_sim.CameraSensorSpec()
        side_rgb.uuid = "side_rgb"
        side_rgb.sensor_type = habitat_sim.SensorType.COLOR
        side_rgb.resolution = [image_size, image_size]
        side_rgb.position = [0.3, 0.6, 0.0]
        agent_cfg.sensor_specifications.append(side_rgb)
        
        # Top-down RGB camera
        top_rgb = habitat_sim.CameraSensorSpec()
        top_rgb.uuid = "top_rgb"
        top_rgb.sensor_type = habitat_sim.SensorType.COLOR
        top_rgb.resolution = [image_size, image_size]
        top_rgb.position = [0.0, 1.5, 0.0]
        agent_cfg.sensor_specifications.append(top_rgb)
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        return sim
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Sample goal position
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        self.goal_position = np.array([x, y, self.goal_height], dtype=np.float32)
        
        # Reset arm to default pose
        self._reset_arm_to_default()
        
        # Clear frame buffers
        for buf in self.frame_buffers:
            buf.clear()
        
        self.current_step = 0
        
        # Get initial observation
        obs = self._get_obs()
        return obs, {}
    
    def _reset_arm_to_default(self):
        """Reset arm to default neutral position"""
        default_pose = np.array([
            0.0,      # shoulder_pan_joint
            1.3,      # shoulder_lift_joint
            0.0,      # upperarm_roll_joint
            -2.2,     # elbow_flex_joint
            0.0,      # forearm_roll_joint
            2.0,      # wrist_flex_joint
            0.0       # wrist_roll_joint
        ], dtype=np.float32)
        self.arm_angles = default_pose
    
    def step(self, action):
        """Execute arm action"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update arm angles
        dt = 0.05  # 50ms timestep
        self.arm_angles = np.clip(
            self.arm_angles + action * dt,
            -np.pi,
            np.pi
        )
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute distance to goal (placeholder - would use forward kinematics)
        gripper_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)  # Placeholder
        to_goal = self.goal_position - gripper_pos
        distance = float(np.linalg.norm(to_goal))
        
        # Reward
        success = distance < self.success_distance
        reward = 1.0 if success else -0.01 * distance
        
        done = success
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance": distance,
            "goal_position": self.goal_position.copy(),
            "arm_angles": self.arm_angles.copy(),
            "success": success
        }
        
        return obs, reward, done, truncated, info
    
    def _get_obs(self):
        """Get vision observation: stacked RGB images from multiple cameras"""
        
        # Get images from each camera
        observations = self.sim.get_sensor_observations()
        
        camera_images = []
        for camera_name in ["front_rgb", "side_rgb", "top_rgb"]:
            if camera_name in observations:
                # Get RGB image (H, W, 3)
                img = observations[camera_name].astype(np.uint8)
                
                # Add to frame buffer (maintaining history)
                # This would be done here for temporal stacking
                camera_images.append(img)
            else:
                # Placeholder if sensor not available
                camera_images.append(
                    np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                )
        
        # Stack images: (num_cameras * channels, image_size, image_size)
        # Concatenate along channel dimension
        stacked_obs = np.concatenate(camera_images, axis=2)  # (H, W, 9)
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))   # (9, H, W)
        
        # For frame stacking (add temporal dimension)
        # In practice, you'd want to maintain frame buffers for each camera
        # For now, return current frame repeated (simplified)
        full_obs = np.repeat(stacked_obs[np.newaxis, :, :, :], self.frame_stack, axis=0)
        full_obs = full_obs.reshape(
            self.frame_stack * self.num_cameras * self.channels_per_camera,
            self.image_size,
            self.image_size
        )
        
        return full_obs.astype(np.uint8)
    
    def close(self):
        self.sim.close()


# Test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING VISION-BASED ARM REACHING ENVIRONMENT")
    print("="*70 + "\n")
    
    try:
        env = VisionArmReachingEnv(image_size=64, frame_stack=4)
        
        print("✓ Environment created!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        obs, info = env.reset()
        print(f"\n✓ Reset successful!")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min()}, {obs.max()}]")
        
        print("\n--- Running episode (10 steps) ---")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step:2d}: dist={info['distance']:.3f}m | reward={reward:.4f}")
            
            if done:
                print("\n✓ GOAL REACHED!")
                break
        
        env.close()
        print("\n" + "="*70)
        print("TEST COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
