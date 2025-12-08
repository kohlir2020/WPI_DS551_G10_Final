"""
Simple GPU Training Script (No Habitat dependency)
Tests GPU training pipeline with basic environment
"""

import os
import sys
import torch
import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from simple_arm_reaching_env import SimpleArmReachingEnv


def main():
    parser = argparse.ArgumentParser(description="Train RL algorithm on arm reaching")
    parser.add_argument("--algorithm", choices=["PPO", "A2C", "SAC"], default="PPO",
                        help="Algorithm to use")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="Device to use")
    args = parser.parse_args()
    
    # Check GPU
    gpu_available = torch.cuda.is_available()
    device = args.device
    if device == "auto":
        device = "cuda" if gpu_available else "cpu"
    
    print(f"\n{'='*70}")
    print(f"GPU Training Setup")
    print(f"{'='*70}")
    print(f"✓ PyTorch CUDA available: {gpu_available}")
    if gpu_available:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Device selected: {device}")
    else:
        print(f"⚠ GPU not available, using CPU")
    print(f"{'='*70}\n")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/simple_arm/{args.algorithm.lower()}_{timestamp}"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"[1/4] Creating Environment")
    env = SimpleArmReachingEnv(max_steps=200)
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    print(f"\n[2/4] Setting up Callbacks")
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix=f"arm_{args.algorithm.lower()}",
        save_replay_buffer=False
    )
    print(f"✓ Checkpoint callback: Save every 10,000 steps")
    
    print(f"\n[3/4] Creating {args.algorithm} Model")
    
    # Algorithm configs
    configs = {
        "PPO": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "ent_coef": 0.01,
            "verbose": 1,
        },
        "A2C": {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 128,
            "ent_coef": 0.01,
            "verbose": 1,
        },
        "SAC": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "ent_coef": "auto",
            "verbose": 1,
        }
    }
    
    config = configs[args.algorithm]
    
    if args.algorithm == "PPO":
        model = PPO(env=env, device=device, tensorboard_log=log_dir, **config)
    elif args.algorithm == "A2C":
        model = A2C(env=env, device=device, tensorboard_log=log_dir, **config)
    elif args.algorithm == "SAC":
        model = SAC(env=env, device=device, tensorboard_log=log_dir, **config)
    
    print(f"✓ Model created with device: {device}")
    print(f"  - Learning rate: {config['learning_rate']}")
    
    print(f"\n[4/4] Training for {args.steps:,} steps")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(log_dir, f"final_{args.algorithm.lower()}")
    model.save(final_path)
    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"  - Final model: {final_path}.zip")
    print(f"  - Logs: {log_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"\nView progress:")
    print(f"  tensorboard --logdir {os.path.dirname(log_dir)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
