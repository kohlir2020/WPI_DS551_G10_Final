"""
IMPROVED HIGH-LEVEL TRAINING
Fixes:
1. Much higher entropy for exploration
2. Better reward shaping  
3. Longer training
4. Better subgoal validation
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from hrl_highlevel_env import HRLHighLevelEnvImproved as HRLHighLevelEnv

# ============================================================
# CONFIGURATION
# ============================================================
LOW_LEVEL_MODEL = "models/lowlevel_curriculum_250k"
SAVE_DIR = "models/hl_improved"
CHECKPOINT_DIR = "models/hl_improved_checkpoints"
LOG_DIR = "./hl_improved_logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# CREATE ENVIRONMENTS
# ============================================================
print("=" * 70)
print("IMPROVED HIGH-LEVEL TRAINING")
print("=" * 70)

print("\nCreating training environment...")
train_env = HRLHighLevelEnv(
    low_level_model_path=LOW_LEVEL_MODEL,
    subgoal_distance=5.0,      # Reduced from 6.0 (more achievable)
    option_horizon=50,         # Increased from 40 (more time to reach)
    debug=False,
)
print("✓ Training environment created")

print("Creating evaluation environment...")
eval_env = HRLHighLevelEnv(
    low_level_model_path=LOW_LEVEL_MODEL,
    subgoal_distance=5.0,
    option_horizon=50,
    debug=False,
)
print("✓ Evaluation environment created")

# ============================================================
# IMPROVED PPO HYPERPARAMETERS
# ============================================================
print("\n" + "=" * 70)
print("HYPERPARAMETERS (IMPROVED)")
print("=" * 70)

hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 512,            # Increased from 128
    "batch_size": 64,
    "gamma": 0.98,             # Higher discount (long-term planning)
    "gae_lambda": 0.95,
    "n_epochs": 10,            # Increased from 4
    "ent_coef": 0.08,          # 8x HIGHER than before! (0.01 → 0.08)
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": LOG_DIR,
    "device": "cpu",
}

print("\nKey Changes:")
print(f"  - Entropy coefficient: {hyperparams['ent_coef']} (was 0.01)")
print(f"    → Much more exploration! Agent will try different directions")
print(f"  - n_steps: {hyperparams['n_steps']} (was 128)")
print(f"  - n_epochs: {hyperparams['n_epochs']} (was 4)")
print(f"  - gamma: {hyperparams['gamma']} (was 0.95)")
print(f"  - Subgoal distance: 5.0m (was 6.0m)")
print(f"  - Option horizon: 50 steps (was 40)")

model = PPO("MlpPolicy", train_env, **hyperparams)

print("\n✓ PPO model created")

# ============================================================
# CALLBACKS
# ============================================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    eval_freq=5000,            # Evaluate every 5k steps
    n_eval_episodes=10,        # More episodes for better estimate
    deterministic=True,
    verbose=1,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="hl_improved",
    verbose=1,
)

# ============================================================
# TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
#before was 500_000
TOTAL_TIMESTEPS = 1_000_000

print(f"\nTotal timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Expected training time: ~30-60 minutes")
print(f"Checkpoints saved every 10k steps to: {CHECKPOINT_DIR}")
print(f"Best model saved to: {SAVE_DIR}")
print(f"\nMonitor training with:")
print(f"  tensorboard --logdir {LOG_DIR}")

print("\n" + "-" * 70)
print("Training started... (this will take a while)")
print("-" * 70 + "\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Save final model
    final_path = f"{SAVE_DIR}/highlevel_improved_final"
    model.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("Saving current model...")
    model.save(f"{SAVE_DIR}/highlevel_improved_interrupted")
    print("✓ Model saved")

# ============================================================
# QUICK EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("QUICK EVALUATION (5 episodes)")
print("=" * 70)

successes = 0
for ep in range(5):
    obs, _ = eval_env.reset()
    done = False
    steps = 0
    
    while not done and steps < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        steps += 1
        
        if done:
            print(f"Episode {ep+1}: ✓ SUCCESS in {steps} steps!")
            successes += 1
            break
        if truncated:
            print(f"Episode {ep+1}: ✗ Timeout (final distance: {info['main_distance']:.1f}m)")
            break

success_rate = (successes / 5) * 100
print(f"\nQuick eval success rate: {success_rate:.0f}% ({successes}/5)")

if success_rate > 0:
    print("✓ System is learning! Some episodes successful!")
else:
    print("⚠️  Still 0% success - may need more training or further tuning")

train_env.close()
eval_env.close()

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"\nNext steps:")
print(f"1. Evaluate with: python evaluate_hl_corrected.py")
print(f"2. Use model: {final_path}.zip")
print(f"3. Check tensorboard: tensorboard --logdir {LOG_DIR}")