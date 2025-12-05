# fetch_navigation_env.py
"""
Navigation environment for Fetch robot in Habitat
Adapted from point-nav to mobile manipulator
"""
import os
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['MAGNUM_LOG'] = 'quiet'
import gymnasium as gym
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
import os
import magnum as mn

class FetchNavigationEnv(gym.Env):
    """Navigation task for Fetch mobile manipulator"""
    
    def __init__(self, scene_path=None):
        super().__init__()
        
        # Use local data directory if available
        if scene_path is None:
            scene_path = os.path.join(
                os.path.dirname(__file__),
                "../../data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
            )
        
        scene_path = os.path.abspath(scene_path)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        
        print(f"Using scene: {scene_path}")
        
        # Create simulator with Fetch robot
        self.sim = self._create_sim(scene_path)
        
        # Get agent (Fetch robot)
        self.agent = self.sim.get_agent(0)
        self.agent = self.sim.get_agent(0)
        print(f"✓ Simulator created!")
        # Action space: [base_forward, base_angular, arm_lift]
        # For now, just navigation (ignore arm)
        self.action_space = gym.spaces.Discrete(4)  # STOP, FWD, LEFT, RIGHT
        
        # Observation: [distance_to_goal, angle_to_goal]
        self.observation_space = gym.spaces.Box(
            low=np.array([-100.0, -np.pi]),
            high=np.array([100.0, np.pi]),
            dtype=np.float32
        )
        
        self.max_steps = 500
        self.current_step = 0
        self.goal_position = None
        
        # Movement parameters (tuned for Fetch)
        self.forward_velocity = 0.25  # m/s
        self.turn_velocity = 10.0  # degrees/s
        self.dt = 0.1  # timestep
        
    def _create_sim(self, scene_path):
        """Create simulator with Fetch robot"""
        
        # Backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = -1
        
        # Agent config - use default agent for now
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Add RGB sensor
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [256, 256]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        return sim
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            
            np.random.seed(seed)
        
        # Reset robot to random navigable position
        pathfinder = self.sim.pathfinder
        
        # Sample positions
        agent_pos = pathfinder.get_random_navigable_point()
        self.goal_position = pathfinder.get_random_navigable_point()
        
        # Ensure goal is at least 2m away
        while np.linalg.norm(agent_pos - self.goal_position) < 2.0:
            self.goal_position = pathfinder.get_random_navigable_point()
        
        # Reset robot state
        self.robot.base_pos = agent_pos
        self.robot.base_rot = 0.0  # Face forward
        
        # Reset arm to default pose
        self._reset_arm()
        
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _reset_arm(self):
        """Reset arm to neutral position"""
        # Fetch default arm configuration
        # [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, 
        #  forearm_roll, wrist_flex, wrist_roll]
        default_arm_pose = np.array([0.0, -0.5, 0.0, 1.5, 0.0, 1.0, 0.0])
        self.robot.arm_joint_pos = default_arm_pose
    
    def step(self, action):
        """Execute action"""
        
        # Map action to base movement
        if action == 1:  # MOVE_FORWARD
            lin_vel = self.forward_velocity
            ang_vel = 0.0
        elif action == 2:  # TURN_LEFT
            lin_vel = 0.0
            ang_vel = self.turn_velocity
        elif action == 3:  # TURN_RIGHT
            lin_vel = 0.0
            ang_vel = -self.turn_velocity
        else:  # STOP
            lin_vel = 0.0
            ang_vel = 0.0
        
        # Apply base control
        self.robot.base_vel = lin_vel
        self.robot.base_rot_vel = np.deg2rad(ang_vel)
        
        # Step simulation
        self.sim.step_world(self.dt)
        
        self.current_step += 1
        
        # Get observation and check success
        obs = self._get_obs()
        distance = obs[0]
        
        done = distance < 0.5  # Success threshold
        reward = 10.0 if done else -0.01
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance": distance,
            "base_pos": self.robot.base_pos.copy(),
        }
        
        return obs, reward, done, truncated, info
    
    def _get_obs(self):
        """Get observation: [distance, angle] to goal"""
        
        # Current base position
        base_pos = self.robot.base_pos
        
        # Distance to goal
        distance = np.linalg.norm(base_pos - self.goal_position)
        
        # Angle to goal (in robot's reference frame)
        goal_direction = self.goal_position - base_pos
        goal_angle = np.arctan2(goal_direction[2], goal_direction[0])
        
        # Robot's heading
        robot_heading = self.robot.base_rot
        
        # Relative angle
        angle = goal_angle - robot_heading
        
        # Normalize to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        return np.array([distance, angle], dtype=np.float32)
    
    def render(self, mode='rgb_array'):
        """Render observation"""
        obs = self.sim.get_sensor_observations()
        return obs.get("rgb", None)
    
    def close(self):
        self.sim.close()


# TEST
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING FETCH NAVIGATION ENVIRONMENT")
    print("="*60 + "\n")
    
    try:
        env = FetchNavigationEnv()
        
        print("✓ Environment created!")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Reset
        obs, _ = env.reset()
        print(f"\n✓ Reset successful!")
        print(f"  Initial distance to goal: {obs[0]:.2f}m")
        print(f"  Initial angle to goal: {np.rad2deg(obs[1]):.1f}°")
        
        # Run random episode
        print("\n--- Running random episode ---")
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            action_names = ["STOP", "FWD", "LEFT", "RIGHT"]
            print(f"Step {step:2d}: {action_names[action]:5s} | "
                  f"dist={info['distance']:6.2f}m | pos={info['base_pos']}")
            
            if done:
                print("\n✓✓✓ SUCCESS! Fetch reached goal! ✓✓✓")
                break
            if truncated:
                print("\n✗ Timeout")
                break
        
        env.close()
        print("\n" + "="*60)
        print("TEST PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Check Fetch URDF exists:")
        print("   ls ~/habitat-lab/data/robots/hab_fetch/robots/hab_fetch.urdf")
        print("2. If missing, download from Habitat documentation")
        print("3. Try with test scene first before ReplicaCAD")
