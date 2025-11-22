import gym
import habitat
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

class NavigationWrapper(gym.Wrapper):
    """
    Improved wrapper with better reward shaping for navigation
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None
        self.steps = 0
        self.max_steps = 500
        self.visited_positions = []
        self.success_count = 0
        
    def reset(self):
        obs = self.env.reset()
        self.steps = 0
        self.visited_positions = []
        
        # Get initial distance to goal
        current_pos = self.env.sim.get_agent_state().position
        goal_pos = self.env.current_episode.goals[0].position
        self.prev_distance = np.linalg.norm(
            np.array(current_pos) - np.array(goal_pos)
        )
        self.initial_distance = self.prev_distance
        
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        
        # Get current position and distance to goal
        current_pos = self.env.sim.get_agent_state().position
        goal_pos = self.env.current_episode.goals[0].position
        current_distance = np.linalg.norm(
            np.array(current_pos) - np.array(goal_pos)
        )
        
        # ============ IMPROVED REWARD SHAPING ============
        
        # 1. HUGE success bonus
        if info.get('success', False):
            reward = 10.0  # Increased from implicit small reward
            done = True
            self.success_count += 1
            print(f"SUCCESS! Total successes: {self.success_count}")
        
        # 2. Strong distance-based progress reward (shaped reward)
        elif self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            
            # Reward getting closer, penalize getting farther
            # Scale by how far we've come overall
            progress_ratio = 1.0 - (current_distance / self.initial_distance)
            reward = distance_delta * 5.0  # Stronger shaping (was implicit ~1.0)
            
            # Bonus for being very close to goal
            if current_distance < 0.5:
                reward += 2.0  # Strong bonus when within 0.5m
            elif current_distance < 1.0:
                reward += 1.0  # Bonus when within 1m
            elif current_distance < 2.0:
                reward += 0.5  # Small bonus within 2m
        
        # 3. Exploration bonus (reward visiting new areas)
        position_tuple = tuple(np.round(current_pos, 1))  # Round to 0.1m grid
        if position_tuple not in self.visited_positions:
            self.visited_positions.append(position_tuple)
            reward += 0.1  # Small exploration bonus
        
        # 4. Small step penalty (encourage efficiency, but not too harsh)
        reward -= 0.01  # Reduced from potential -0.05
        
        # 5. Collision penalty (if colliding)
        if info.get('collisions', {}).get('is_collision', False):
            reward -= 0.5  # Penalize collisions
        
        # 6. Timeout handling
        if self.steps >= self.max_steps:
            done = True
            # Reduced timeout penalty, add partial credit for progress
            progress_reward = (1.0 - current_distance / self.initial_distance) * 2.0
            reward += progress_reward - 1.0  # Partial credit instead of harsh penalty
        
        self.prev_distance = current_distance
        
        return obs, reward, done, info

def make_env():
    """Create and wrap the Habitat environment"""
    from omegaconf import OmegaConf
    
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    
    # Make config writable (OmegaConf struct mode)
    OmegaConf.set_struct(config, False)
    
    # Modify config
    config.habitat.dataset.data_path = "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
    config.habitat.simulator.scene = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    config.habitat.task.success_distance = 0.2  # 20cm to goal = success
    config.habitat.environment.max_episode_steps = 500
    
    # Make it read-only again (optional, for safety)
    OmegaConf.set_struct(config, True)
    
    env = habitat.Env(config=config)
    env = NavigationWrapper(env)
    return env

def train_agent():
    """Train PPO agent with improved hyperparameters"""
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # ============ IMPROVED PPO HYPERPARAMETERS ============
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # Learning parameters
        learning_rate=3e-4,        # Standard LR
        n_steps=2048,              # More steps per update (was default 2048)
        batch_size=64,             # Smaller batches for better gradients
        n_epochs=10,               # Standard epochs
        gamma=0.99,                # Standard discount factor
        
        # Exploration parameters (CRITICAL FOR NAVIGATION)
        ent_coef=0.01,             # Increased entropy (more exploration)
        clip_range=0.2,            # Standard PPO clip
        
        # Value function parameters
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        
        # Policy network architecture
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network
        ),
        
        tensorboard_log="./navigation_tensorboard/",
    )
    
    # Checkpoint callback - save every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/",
        name_prefix="nav_model"
    )
    
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Total timesteps: 2,000,000")
    print(f"Entropy coefficient: 0.01 (INCREASED for exploration)")
    print(f"Network architecture: 256x256 (larger)")
    print(f"Success distance: 0.2m")
    print(f"Max episode steps: 500")
    print("=" * 60)
    print("\nReward Structure:")
    print("  - Success: +10.0 (HUGE bonus)")
    print("  - Progress to goal: +5.0 per meter closer")
    print("  - Within 0.5m: +2.0 bonus")
    print("  - Within 1.0m: +1.0 bonus")
    print("  - Exploration: +0.1 per new cell")
    print("  - Step penalty: -0.01")
    print("  - Collision: -0.5")
    print("  - Timeout with progress: partial credit")
    print("=" * 60)
    print("\nStarting training...\n")
    
    # Train the agent
    model.learn(
        total_timesteps=2_000_000,  # 2M steps
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("navigation_agent_final")
    print("\n" + "=" * 60)
    print("Training complete! Model saved as 'navigation_agent_final.zip'")
    print("=" * 60)
    
    return model, env

def evaluate_agent(model, env, n_episodes=20):
    """Evaluate trained agent"""
    print("\n" + "=" * 60)
    print("EVALUATING TRAINED POLICY")
    print("=" * 60)
    
    successes = 0
    total_steps = []
    distances = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Get final distance
        env_single = env.envs[0]
        current_pos = env_single.env.sim.get_agent_state().position
        goal_pos = env_single.env.current_episode.goals[0].position
        final_distance = np.linalg.norm(
            np.array(current_pos) - np.array(goal_pos)
        )
        
        success = final_distance < 0.2
        if success:
            successes += 1
            total_steps.append(steps)
            print(f"Ep {ep+1}: ✓ SUCCESS in {steps} steps")
        else:
            distances.append(final_distance)
            print(f"Ep {ep+1}: ✗ TIMEOUT (still {final_distance:.2f}m away)")
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    success_rate = (successes / n_episodes) * 100
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{n_episodes})")
    
    if total_steps:
        avg_steps = np.mean(total_steps)
        print(f"Avg Steps to Goal: {avg_steps:.1f}")
    else:
        print(f"Avg Steps to Goal: N/A (no successes)")
    
    if distances:
        avg_distance = np.mean(distances)
        print(f"Avg Distance on Failure: {avg_distance:.2f}m")
    
    # Compare to random baseline (~10% success)
    improvement = success_rate - 10.0
    print(f"Improvement over Random: {improvement:+.1f}%")
    print("=" * 60)
    
    # Save results
    with open("training_results.txt", "w") as f:
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Successes: {successes}/{n_episodes}\n")
        f.write(f"Avg Steps: {np.mean(total_steps) if total_steps else 0:.1f}\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
    
    print("\nResults saved to training_results.txt")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HABITAT NAVIGATION TRAINING - IMPROVED VERSION")
    print("=" * 60)
    
    # Train agent
    model, env = train_agent()
    
    # Evaluate agent
    evaluate_agent(model, env, n_episodes=20)
    
    env.close()
    print("\nDone!")