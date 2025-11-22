# train_nav_safe.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from simple_navigation_env import SimpleNavigationEnv
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = SimpleNavigationEnv()

# Auto-save every 10k steps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./models/checkpoints/",
    name_prefix="nav_checkpoint",
)

print("Training with auto-checkpoints every 10k steps")
print("Safe to stop anytime - progress is saved!\n")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    device="cpu",
)

try:
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    model.save("models/nav_ppo_100k_final")
    print("\n✓ Training complete!")
    
except KeyboardInterrupt:
    print("\n⚠ Stopped by user - saving...")
    model.save("models/nav_ppo_interrupted")

env.close()