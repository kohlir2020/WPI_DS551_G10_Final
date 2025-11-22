# simple_navigation_env.py (FIXED)
import gymnasium as gym
import numpy as np
import habitat_sim
import os

class SimpleNavigationEnv(gym.Env):
    """Direct habitat-sim navigation"""
    
    def __init__(self):
        super().__init__()
        
        scene_path = "/home/pinaka/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        
        print(f"Using scene: {scene_path}")
        
        # Create simulator
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False
        backend_cfg.gpu_device_id = -1
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        
        # Get pathfinder for navigable points
        self.pathfinder = self.sim.pathfinder
        
        print("✓ Simulator created!")
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=np.array([-100.0, -np.pi]),
            high=np.array([100.0, np.pi]),
            dtype=np.float32
        )
        
        self.max_steps = 500
        self.current_step = 0
        self.goal_position = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Sample navigable points
        agent_pos = self.pathfinder.get_random_navigable_point()
        self.goal_position = self.pathfinder.get_random_navigable_point()
        
        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = agent_pos
        self.agent.set_state(agent_state)
        
        self.current_step = 0
        return self._get_obs(), {}
    
    def step(self, action):
        if action == 1:  # FORWARD
            self.agent.act("move_forward")
        elif action == 2:  # LEFT
            self.agent.act("turn_left")
        elif action == 3:  # RIGHT
            self.agent.act("turn_right")
        
        self.current_step += 1
        obs = self._get_obs()
        distance = obs[0]
        
        done = distance < 0.5
        reward = 10.0 if done else -0.01
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, done, truncated, {"distance": distance}
    
    def _get_obs(self):
        agent_state = self.agent.get_state()
        agent_pos = agent_state.position
        
        distance = np.linalg.norm(agent_pos - self.goal_position)
        
        direction = self.goal_position - agent_pos
        angle = np.arctan2(direction[2], direction[0])
        
        return np.array([distance, angle], dtype=np.float32)
    
    def close(self):
        self.sim.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING NAVIGATION")
    print("="*60 + "\n")
    
    try:
        env = SimpleNavigationEnv()
        obs, _ = env.reset()
        print(f"✓ Reset! Distance: {obs[0]:.2f}m\n")
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            names = ["STOP", "FWD", "LEFT", "RIGHT"]
            print(f"Step {step:2d}: {names[action]:5s} | {info['distance']:6.2f}m")
            
            if done:
                print("\n✓ SUCCESS!")
                break
        
        env.close()
        print("\n" + "="*60)
        print("TEST PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()