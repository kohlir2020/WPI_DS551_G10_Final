#!/usr/bin/env python3
"""
Test training script for Docker - verifies Habitat + RL working
"""
import sys
sys.path.insert(0, '/workspace/src/arm')

from habitat_arm_reaching_env import HabitatArmReachingEnv
from stable_baselines3 import SAC
import torch

def test_docker_training():
    print("\n" + "="*70)
    print("HABITAT DOCKER TRAINING TEST")
    print("="*70)
    
    # Check CUDA
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
    
    # Create environment
    print("\n[1/5] Creating HabitatArmReachingEnv...")
    env = HabitatArmReachingEnv()
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    # Reset environment
    print("\n[2/5] Testing environment reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful. Obs shape: {obs.shape}")
    
    # Test step
    print("\n[3/5] Testing environment step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful. Reward: {reward:.4f}")
    
    # Create model
    print("\n[4/5] Creating SAC model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        device=device,
        verbose=0
    )
    print(f"✓ Model created on device: {device}")
    
    # Train for short period
    print("\n[5/5] Training for 1000 steps...")
    model.learn(total_timesteps=1000)
    print(f"✓ Training completed successfully!")
    
    # Cleanup
    env.close()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - HABITAT DOCKER READY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_docker_training()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
