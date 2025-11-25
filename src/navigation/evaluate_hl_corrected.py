"""
CORRECTED High-Level Evaluation Script
Fixes the issue where wrong model was being loaded
"""
from hrl_highlevel_env import HRLHighLevelEnvImproved as HRLHighLevelEnv
from stable_baselines3 import PPO
import numpy as np

# ============================================================
# CRITICAL FIX: Load correct models
# ============================================================
LOW_LEVEL_MODEL_PATH = "models/lowlevel_curriculum_250k"  # ✓ Correct!
HIGH_LEVEL_MODEL_PATH = "models/hl_improved/highlevel_improved_final"  # ✓ Correct!

print("=" * 70)
print("HRL HIGH-LEVEL EVALUATION (CORRECTED)")
print("=" * 70)

# ============================================================
# Create environment with CORRECT low-level model
# ============================================================
print("\nCreating environment...")
env = HRLHighLevelEnv(
    low_level_model_path=LOW_LEVEL_MODEL_PATH,  # ✓ Use trained low-level skill
    subgoal_distance=6.0,
    option_horizon=40,
    debug=True  # Enable debugging output
)
print("✓ Environment created")

# ============================================================
# Load HIGH-level policy
# ============================================================
print(f"\nLoading high-level policy from: {HIGH_LEVEL_MODEL_PATH}")
model = PPO.load(HIGH_LEVEL_MODEL_PATH)
print("✓ High-level policy loaded")

# ============================================================
# Run evaluation
# ============================================================
n_episodes = 20
successes = 0
total_steps = []
final_distances = []

print(f"\nEvaluating for {n_episodes} episodes...\n")
print("=" * 70)

for ep in range(n_episodes):
    obs, _ = env.reset()
    main_goal = env.main_goal
    agent_start = env.ll_env.agent.get_state().position
    start_distance = np.linalg.norm(agent_start - main_goal)
    
    print(f"\nEPISODE {ep + 1}/{n_episodes}")
    print("-" * 70)
    print(f"Start Position : {agent_start}")
    print(f"Main Goal      : {main_goal}")
    print(f"Start Distance : {start_distance:.2f}m\n")
    
    episode_done = False
    step_count = 0
    
    for step in range(20):
        step_count += 1
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        print(f"HL Step {step+1}:")
        print(f"  Observation: dist={obs[0]:.2f}m, angle={obs[1]:.2f}rad")
        print(f"  Action (direction): {action}")
        
        # Execute
        agent_pos_before = env.ll_env.agent.get_state().position
        obs, reward, done, truncated, info = env.step(action)
        agent_pos_after = env.ll_env.agent.get_state().position
        
        movement = np.linalg.norm(agent_pos_after - agent_pos_before)
        
        print(f"  Movement: {movement:.3f}m")
        print(f"  Distance to goal: {info['main_distance']:.2f}m")
        print(f"  Reward: {reward:.3f}")
        
        if done:
            print(f"\n✓✓✓ MAIN GOAL REACHED in {step_count} steps! ✓✓✓")
            successes += 1
            total_steps.append(step_count)
            final_distances.append(info['main_distance'])
            episode_done = True
            break
            
        if truncated:
            print(f"\n✗ MAX HL STEPS REACHED (still {info['main_distance']:.2f}m away)")
            final_distances.append(info['main_distance'])
            episode_done = True
            break
    
    if not episode_done:
        print(f"\n✗ Episode timeout at {info['main_distance']:.2f}m from goal")
        final_distances.append(info['main_distance'])

env.close()

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

success_rate = (successes / n_episodes) * 100
avg_steps = np.mean(total_steps) if total_steps else 0
avg_final_dist = np.mean(final_distances)

print(f"\nSuccess Rate    : {success_rate:.1f}% ({successes}/{n_episodes})")
print(f"Average Steps   : {avg_steps:.1f}")
print(f"Avg Final Dist  : {avg_final_dist:.2f}m")

if success_rate == 0:
    print("\n⚠️  WARNING: 0% success rate!")
    print("   Possible issues:")
    print("   1. High-level policy didn't learn (needs retraining)")
    print("   2. Subgoal generation is broken")
    print("   3. Low-level skill not executing properly")
    print("   4. Action space collapsed (always same action)")
else:
    print(f"\n✓ System is working! {successes}/{n_episodes} episodes successful")

print("=" * 70)