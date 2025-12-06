"""
Fetch Arm Reaching Environment
Train arm to reach toward objects (e.g., drawer handles, fridge)
Gripper triggers automatically when within 15cm of goal
No gripper training - only arm reaching
"""

import gymnasium as gym
import numpy as np
import habitat_sim
# from habitat.articulated_agents.robots.fetch_robot import FetchRobot
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


class FetchArmReachingEnv(gym.Env):
    """
    Fetch arm reaching environment.
    Assumes base has already navigated to correct position.
    Task: Reach arm toward goal position (e.g., drawer handle at 15cm height)
    Success: Gripper within 15cm of goal (triggers gripper automatically)
    
    Observation: [distance_to_goal, arm_joint_angles(7)]
    Actions: Continuous arm joint velocities (7-DOF)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, 
                 scene_path=None,
                 goal_height=0.5,  # Height of drawer handle (meters)
                 success_distance=0.15,  # Gripper activation distance (meters)
                 max_steps=200):
        """
        Args:
            scene_path: Path to scene (uses local data if None)
            goal_height: Height of goal (drawer/fridge handle) in meters
            success_distance: Distance threshold for gripper trigger (meters)
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        if scene_path is None:
            scene_path = os.path.join(
                os.path.dirname(__file__),
                "../../habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
            )
        
        scene_path = os.path.abspath(scene_path)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        
        print(f"Using scene: {scene_path}")
        
        # Initialize simulator
        self.sim = self._create_sim(scene_path)
        print("✓ Simulator initialized")
        
        # Goal parameters
        self.goal_height = goal_height
        self.success_distance = success_distance
        self.max_steps = max_steps
        self.current_step = 0
        self.goal_position = None
        
        # Arm parameters (Fetch has 7-DOF arm)
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
        
        # Action space: continuous velocities for 7 arm joints
        self.action_space = gym.spaces.Box(
            low=-1.0,  # rad/s
            high=1.0,
            shape=(self.num_arm_joints,),
            dtype=np.float32
        )
        
        # Observation space: [distance_to_goal, arm_angles (7)]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] + [-np.pi]*7, dtype=np.float32),
            high=np.array([5.0] + [np.pi]*7, dtype=np.float32),
            dtype=np.float32
        )
        
        print(f"✓ Arm reaching environment initialized")
        print(f"  - Goal height: {goal_height}m")
        print(f"  - Success distance: {success_distance}m")
        print(f"  - Max steps: {max_steps}")
    
    def _create_sim(self, scene_path):
        """Create simulator with Fetch robot"""
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = -1
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = []
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        # Load Fetch Robot
        ao_mgr = sim.get_articulated_object_manager()
        urdf_path = "habitat-lab/data/robots/hab_fetch/robots/hab_fetch.urdf"
        urdf_path = os.path.abspath(urdf_path)
        if not os.path.exists(urdf_path):
             # Try relative path if absolute fails or assume it's correct
             pass
             
        self.robot = ao_mgr.add_articulated_object_from_urdf(urdf_path, fixed_base=True)
        self.robot.translation = np.array([0.0, 0.0, 0.0])
        
        return sim

    def check_collision(self, arm_joint_angles):
        """
        Check if the robot is in collision at the given arm joint angles.
        Args:
            arm_joint_angles: np.array of shape (7,)
        Returns:
            bool: True if in collision, False otherwise
        """
        # Save current state
        current_joint_pos = self.robot.joint_positions
        
        # Create new state
        new_joint_pos = current_joint_pos.copy()
        # Arm joints are indices 5 to 11 (inclusive)
        # Ensure arm_joint_angles is correct shape
        if len(arm_joint_angles) != 7:
            raise ValueError(f"Expected 7 arm joint angles, got {len(arm_joint_angles)}")
            
        new_joint_pos[5:12] = arm_joint_angles
        
        # Set new state
        self.robot.joint_positions = new_joint_pos
        
        # Update collision world
        self.sim.perform_discrete_collision_detection()
        
        # Check contacts
        contacts = self.sim.get_physics_contact_points()
        
        # Restore state
        self.robot.joint_positions = current_joint_pos
        
        # Filter contacts to see if robot is involved
        # If contacts is not empty, there is a collision in the scene.
        # Since the scene is static (except robot), any contact implies robot collision 
        # (either self-collision or collision with environment).
        
        return len(contacts) > 0
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Sample goal position: random x,y on ground, fixed height
        x = np.random.uniform(-1.0, 1.0)  # Relative to base
        y = np.random.uniform(-1.0, 1.0)
        self.goal_position = np.array([x, y, self.goal_height], dtype=np.float32)
        
        # Reset arm to default pose
        self._reset_arm_to_default()
        
        self.current_step = 0
        return self._get_obs(), {}
    
    def _reset_arm_to_default(self):
        """Reset arm to default neutral position"""
        # Fetch default arm configuration (7 joints)
        # Values from Fetch documentation
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
        
        # Apply to robot in simulator
        current_joint_pos = self.robot.joint_positions
        # Arm joints are indices 5 to 11
        current_joint_pos[5:12] = default_pose
        self.robot.joint_positions = current_joint_pos
    
    def step(self, action):
        """Execute arm action"""
        # action: continuous velocities for 7 arm joints
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update arm angles (simple integration)
        dt = 0.05  # 50ms timestep
        self.arm_angles = np.clip(
            self.arm_angles + action * dt,
            -np.pi,
            np.pi
        )
        
        # Apply to robot in simulator
        current_joint_pos = self.robot.joint_positions
        current_joint_pos[5:12] = self.arm_angles
        self.robot.joint_positions = current_joint_pos
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_obs()
        distance = obs[0]
        
        # Check success: gripper within 15cm of goal
        success = distance < self.success_distance
        reward = 1.0 if success else -0.01 * distance  # Reward proximity
        
        done = success
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance": float(distance),
            "goal_position": self.goal_position.copy(),
            "arm_angles": self.arm_angles.copy(),
            "success": success
        }
        
        return obs, reward, done, truncated, info
    
    def _get_obs(self):
        """Get observation: [distance_to_goal, arm_angles]"""
        # Get gripper position (end effector)
        # Gripper link is index 22 (gripper_link)
        # We need to get the link ID from the robot
        link_ids = self.robot.get_link_ids()
        # Assuming index 22 corresponds to gripper_link as verified in inspection
        gripper_link_id = link_ids[22]
        
        gripper_pos = self.robot.get_link_scene_node(gripper_link_id).translation
        
        # Distance to goal
        to_goal = self.goal_position - gripper_pos
        distance = float(np.linalg.norm(to_goal))
        
        # Observation
        obs = np.concatenate([
            np.array([distance], dtype=np.float32),
            self.arm_angles.astype(np.float32)
        ])
        
        return obs
    
    def close(self):
        self.sim.close()


# Simple test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING FETCH ARM REACHING ENVIRONMENT")
    print("="*70 + "\n")
    
    try:
        env = FetchArmReachingEnv()
        
        print("✓ Environment created!")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Reset
        obs, info = env.reset()
        print(f"\n✓ Reset successful!")
        print(f"  Initial obs shape: {obs.shape}")
        print(f"  Distance to goal: {obs[0]:.3f}m")
        
        # Run episode
        print("\n--- Running episode (10 steps) ---")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step:2d}: dist={info['distance']:.3f}m | "
                  f"reward={reward:.4f} | success={info['success']}")
            
            if done:
                print("\n✓ GOAL REACHED - Gripper triggered!")
                break
        
        env.close()
        print("\n" + "="*70)
        print("TEST COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
