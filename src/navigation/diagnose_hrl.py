"""
We use this diagnostic script to identify why high-level HRL is failing
"""
import numpy as np
from stable_baselines3 import PPO
import sys
import os

# 1.We check if models exist
ll_model_path = "models/lowlevel_curriculum_1M"
hl_model_path = "models/hl_improved_2M/highlevel_improved_final_2M"

if os.path.exists(f"{ll_model_path}.zip"):
    try:
        ll_model = PPO.load(ll_model_path)
        print(f"Successfully loaded low-level PPO model,Policy network: {ll_model.policy}")
    except Exception as e:
        print(f"Error loading low-level model: {e}")
        sys.exit(1)
else:
    print(f"Low-level model NOT found: {ll_model_path}.zip")
    sys.exit(1)

if os.path.exists(f"{hl_model_path}.zip"):
    try:
        hl_model = PPO.load(hl_model_path)
        print(f"Successfully loaded high-level PPO model, Policy network: {hl_model.policy}")
    except Exception as e:
        print(f"Error loading high-level model: {e}")
        sys.exit(1)
else:
    print(f"High-level model NOT found: {hl_model_path}.zip")
    sys.exit(1)

# 2) We test low-level environment
try:
    from simple_navigation_env import SimpleNavigationEnv
    ll_env = SimpleNavigationEnv()

    # Reset and check observation
    obs, info = ll_env.reset()
    print(f"Observation shape: {obs.shape}, Observation: {obs}, Initial distance to goal: {obs[0]:.2f}m,Initial angle to goal: {obs[1]:.2f} rad")

    # Test a few steps
    print("Testing low-level policy")
    for i in range(10):
        action, _ = ll_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = ll_env.step(action)
        print(f"Step {i+1}: action={action}, dist={obs[0]:.2f}m, reward={reward:.3f}")
        if done or truncated:
            print(f"Episode ended: done={done}, truncated={truncated}")
            break
    
    ll_env.close()
    print("Low-level environment works correctly")
    
except Exception as e:
    print(f"Error with low-level environment:{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3) Test high-level environment setup
try:
    from hrl_highlevel_env import HRLHighLevelEnv
    # Create high-level environment
    hl_env = HRLHighLevelEnv(
        low_level_model_path=ll_model_path,
        subgoal_distance=6.0,
        option_horizon=40,
        debug=False
    )    
    # Reset
    obs, info = hl_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Initial distance to main goal: {obs[0]:.2f}m")
    print(f"Initial angle to main goal: {obs[1]:.2f} rad")
    
    agent_pos = hl_env.ll_env.agent.get_state().position
    main_goal = hl_env.main_goal
    print(f"Agent position: {agent_pos}")
    print(f"Main goal: {main_goal}")
    print(f"Actual distance: {np.linalg.norm(agent_pos - main_goal):.2f}m")
    
except Exception as e:
    print(f"Error creating high-level environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4 Test high-level policy predictions
print("Testing policy predictions for different observations:")
test_observations = [
    np.array([15.0, 0.0], dtype=np.float32),# Goal ahead
    np.array([15.0, np.pi/2], dtype=np.float32),# Goal to the right
    np.array([15.0, -np.pi/2], dtype=np.float32),# Goal to the left
    np.array([15.0, np.pi], dtype=np.float32),# Goal behind
    np.array([5.0, 0.0], dtype=np.float32),# Close goal ahead
]

for i, test_obs in enumerate(test_observations):
    action, _ = hl_model.predict(test_obs, deterministic=True)
    action_probs = hl_model.policy.get_distribution(hl_model.policy.obs_to_tensor(test_obs)[0]).distribution.probs.detach().cpu().numpy()
    print(f"\nTest {i+1}: obs={test_obs},Predicted action: {action},Action probabilities: {action_probs}")
    print(f"Max prob: {action_probs.max():.3f}, Entropy: {-np.sum(action_probs * np.log(action_probs + 1e-10)):.3f}")

# To check if policy is deterministic (collapsed)
if action_probs.max() > 0.95:
    print("\nWARNING: Policy is nearly deterministic (collapsed)!")

# 5 Test actual high-level execution
obs, _ = hl_env.reset()
print(f"Starting position: {hl_env.ll_env.agent.get_state().position}")
print(f"Main goal: {hl_env.main_goal},Initial distance: {obs[0]:.2f}m")

print("\nExecuting 5 high-level steps:")
for step in range(5):
    action, _ = hl_model.predict(obs, deterministic=True)
    print(f"\nHL Step {step+1}:,Current obs: {obs},Chosen action: {action}")

    agent_pos_before = hl_env.ll_env.agent.get_state().position
    obs, reward, done, truncated, info = hl_env.step(action)
    agent_pos_after = hl_env.ll_env.agent.get_state().position
    movement = np.linalg.norm(agent_pos_after - agent_pos_before)
    
    print(f"Agent moved: {movement:.3f}m")
    print(f"New distance to goal: {info['main_distance']:.2f}m, Reward: {reward:.3f}")
    
    if movement < 0.1:
        print("WARNING: Agent barely moved!")
    if done:
        print("Reached main goal!")
        break
    if truncated:
        print("Episode truncated")
        break

hl_env.close()