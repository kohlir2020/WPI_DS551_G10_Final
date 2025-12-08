"""
Train RL policies in Habitat arm reaching environment
Compatible with both Habitat simulator and fallback realistic simulation
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from habitat_arm_reaching_env import HabitatArmReachingEnv
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

def train_habitat_policy(algorithm='PPO', steps=100000, device='cuda', use_habitat=False):
    """Train RL policy in Habitat-compatible environment"""
    
    env_type = "Habitat" if use_habitat else "Realistic Simulation"
    print("\n" + "="*70)
    print(f"ARM REACHING ({env_type}) - TRAINING {algorithm}")
    print("="*70)
    print(f"Environment: {env_type}")
    print(f"Scene: apartment_1")
    print(f"Steps: {steps:,}")
    print(f"Device: {device}")
    
    try:
        # Create environment
        env = HabitatArmReachingEnv(
            max_steps=200,
            scene="apartment_1",
            use_rgb_d=False,
            use_habitat=use_habitat
        )
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = "habitat" if use_habitat else "realistic"
        log_dir = f"logs/simple_arm/{env_name}_{algorithm.lower()}_{timestamp}"
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
            progress_bar=False
        )
        
        # Save final model
        final_path = f"{log_dir}/final_{algorithm.lower()}.zip"
        model.save(final_path)
        
        print(f"\n✅ Training complete!")
        print(f"Final model saved: {final_path}")
        print(f"Tensorboard logs: tensorboard --logdir {log_dir}")
        
        env.close()
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL policy in Habitat-compatible environment')
    parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'A2C', 'SAC'],
                        help='RL algorithm to use')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--habitat', action='store_true',
                        help='Use real Habitat if available (default: realistic simulation)')
    
    args = parser.parse_args()
    
    train_habitat_policy(
        algorithm=args.algorithm,
        steps=args.steps,
        device=args.device,
        use_habitat=args.habitat
    )

