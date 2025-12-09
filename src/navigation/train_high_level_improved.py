"""IMPROVED HIGH-LEVEL TRAINING
Fixes that we did:
1.Much higher entropy for exploration
2.Better reward shaping  
3.Longer training
4.Better subgoal validation"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import sys
# Add current directory to path for imports
sys.path.insert(0, os.getcwd())
from hrl_highlevel_env import HRLHighLevelEnvImproved as HRLHighLevelEnv

# Configuration
LOW_LEVEL_MODEL = "models/lowlevel_curriculum_1M"
SAVE_DIR = "models/hl_improved_2M"
CHECKPOINT_DIR = "models/hl_improved_checkpoints_2M"
LOG_DIR = "./hl_improved_2M_logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create training environment
train_env = HRLHighLevelEnv(
    low_level_model_path=LOW_LEVEL_MODEL,
    subgoal_distance=5.0, 
    option_horizon=50,  
    debug=False,
)
print("Training environment created")

# Create evaluation environment
eval_env = HRLHighLevelEnv(
    low_level_model_path=LOW_LEVEL_MODEL,
    subgoal_distance=5.0,
    option_horizon=50,
    debug=False,
)
print("Evaluation environment created")

# Improved PPO hyperparameters
hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "gamma": 0.90,
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "ent_coef": 0.08,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": LOG_DIR,
    "device": "cpu",
}

print("\nKey Changes:")
print(f" Entropy coefficient: {hyperparams['ent_coef']} (was 0.01)")
print(f" Much more exploration! Agent will try different directions")
print(f" n_steps: {hyperparams['n_steps']} (was 128)")
print(f" n_epochs: {hyperparams['n_epochs']} (was 4)")
print(f" gamma: {hyperparams['gamma']} (was 0.95)")


model = PPO("MlpPolicy", train_env, **hyperparams)
print("\nPPO model created")

# Callbacks for evaluation and checkpointing
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    eval_freq=8000,
    n_eval_episodes=10, # More episodes for better estimate
    deterministic=True,
    verbose=1,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=CHECKPOINT_DIR,
    name_prefix="hl_improved",
    verbose=1,
)

# Training
TOTAL_TIMESTEPS = 2_000_000

print(f"\nTotal timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Best model saved to: {SAVE_DIR}")
print(f"  tensorboard --logdir {LOG_DIR}")
print("Training started")
print("==================================== \n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    print("TRAINING COMPLETE!")
    print("====================================")
    
    # Save final model
    final_path = f"{SAVE_DIR}/highlevel_improved_final_2M"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user \n Saving current model...")
    model.save(f"{SAVE_DIR}/highlevel_improved_interrupted_2M")
    print("Model saved")

# QUICK EVALUATION
print("QUICK EVALUATION (5 episodes)")
print("====================================")

successes = 0
for ep in range(5):
    obs, _ = eval_env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        steps += 1
        if done:
            print(f"Episode {ep+1}: SUCCESS in {steps} steps!")
            successes += 1
            break
        if truncated:
            print(f"Episode {ep+1}: Timeout (final distance: {info['main_distance']:.1f}m)")
            break

success_rate = (successes / 5) * 100
print(f"\nQuick eval success rate: {success_rate:.0f}% ({successes}/5)")

if success_rate > 0:
    print(" System is learning some episodes successful!")
else:
    print(" Still 0% success - may need more training")

train_env.close()
eval_env.close()

print("====================================")
print("DONE now we can evaluate the trained model properly!")
print(f"1. Evaluate with: python evaluate_hl_corrected.py")
print(f"2. Use model: {final_path}.zip")
print(f"3. Check tensorboard: tensorboard --logdir {LOG_DIR}")