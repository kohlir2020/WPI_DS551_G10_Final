"""
Train RL policies on Cartesian (end-effector position) control
Uses same hyperparameters as Phase 1 for fair comparison
"""

import os
import argparse
from datetime import datetime
import sys

# Add src/arm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cartesian_arm_reaching_env import CartesianArmReachingEnv
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

def train_cartesian_policy(algorithm='PPO', steps=100000, device='cuda'):
    """Train RL policy on Cartesian control environment"""
    
    print("\n" + "="*70)
    print(f"PHASE 2: TRAINING {algorithm} WITH CARTESIAN CONTROL")
    print("="*70)
    print(f"Environment: CartesianArmReachingEnv (3D EE position control)")
    print(f"Steps: {steps:,}")
    print(f"Device: {device}")
    
    # Create environment
    env = CartesianArmReachingEnv(max_steps=200)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/simple_arm/cartesian_{algorithm.lower()}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\nLog directory: {log_dir}")
    
    # Configure device
    device_map = torch.device(device)
    
    # Algorithm-specific hyperparameters
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=f"{log_dir}/PPO_1",
            device=device_map
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy',
            env,
            learning_rate=7e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{log_dir}/A2C_1",
            device=device_map
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef='auto',
            target_update_interval=1,
            verbose=1,
            tensorboard_log=f"{log_dir}/SAC_1",
            device=device_map
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Checkpoint callback (every 10k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix=f"arm_{algorithm.lower()}"
    )
    
    print(f"\n🚀 Starting training...")
    
    # Train
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_path = f"{log_dir}/final_{algorithm.lower()}.zip"
    model.save(final_path)
    
    print(f"\n✅ Training complete!")
    print(f"Final model saved: {final_path}")
    print(f"Tensorboard logs: tensorboard --logdir {log_dir}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL policy with Cartesian control')
    parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'A2C', 'SAC'],
                        help='RL algorithm to use')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to train on')
    
    args = parser.parse_args()
    
    train_cartesian_policy(algorithm=args.algorithm, steps=args.steps, device=args.device)
