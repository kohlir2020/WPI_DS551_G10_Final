from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from simple_navigation_env import SimpleNavigationEnv
import os

# ==========================================
# 1. Setup folders
# ==========================================
os.makedirs("models", exist_ok=True)
os.makedirs("logs_lowlevel", exist_ok=True)

# ==========================================
# 2. Create env
# ==========================================
env = SimpleNavigationEnv()
eval_env = SimpleNavigationEnv()

print("Checking env with SB3...")
check_env(env, warn=True)

# ==========================================
# 3. PPO hyperparameters (tuned for local nav)
# ==========================================
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    gamma=0.90,          # shorter horizon
    gae_lambda=0.90,
    clip_range=0.2,
    ent_coef=0.005,      # a bit more exploration
    n_epochs=4,
    tensorboard_log="./logs_lowlevel/",
    verbose=1,
    device="cpu",
)

# ==========================================
# 4. Callbacks
# ==========================================
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path="./models/lowlevel_checkpoints/",
    name_prefix="lowlevel_nav",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/lowlevel_best/",
    n_eval_episodes=10,
    eval_freq=20_000,
    deterministic=True,
    verbose=1,
)

# ==========================================
# 5. Train
# ==========================================
print("\nStarting LOW-LEVEL PPO training (curriculum goals)...\n")

model.learn(
    total_timesteps=250_000,
    callback=[checkpoint_callback, eval_callback],
    reset_num_timesteps=True,
)

model.save("models/lowlevel_curriculum_250k")

env.close()
eval_env.close()

print("\n✓ Low-level training DONE. Model: models/lowlevel_curriculum_250k\n")
