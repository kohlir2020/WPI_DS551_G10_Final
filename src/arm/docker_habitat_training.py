#!/usr/bin/env python3
"""
Full training test in Docker - Habitat + SAC for 10k steps
"""
import sys
sys.path.insert(0, '/workspace/src/arm')

from habitat_arm_reaching_env import HabitatArmReachingEnv
from stable_baselines3 import SAC
import torch

def train_sac_habitat():
    print("\n" + "="*80)
    print("FULL HABITAT SAC TRAINING IN DOCKER")
    print("="*80)
    
    # Environment setup
    print("\n[1/3] Setting up Habitat environment...")
    env = HabitatArmReachingEnv()
    print(f"✓ Environment: {env.__class__.__name__}")
    print(f"✓ Obs space: {env.observation_space}")
    print(f"✓ Act space: {env.action_space}")
    
    # Model creation
    print("\n[2/3] Creating SAC model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        buffer_size=50000,
        device=device,
        verbose=1,
        tensorboard_log="/workspace/logs/docker_test_sac"
    )
    print(f"✓ SAC model created on device: {device}")
    
    # Training
    print("\n[3/3] Training for 10,000 steps...")
    print("(This will show training progress)\n")
    
    model.learn(total_timesteps=10000)
    
    print("\n" + "="*80)
    print("✅ FULL TRAINING SUCCESSFUL!")
    print("="*80 + "\n")
    
    env.close()

if __name__ == "__main__":
    try:
        train_sac_habitat()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
