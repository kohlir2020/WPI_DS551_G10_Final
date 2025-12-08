#!/usr/bin/env python3
"""
GPU Training Test - Force CUDA
"""
import sys
import os
sys.path.insert(0, '/workspace/src/arm')

# Force CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from habitat_arm_reaching_env import HabitatArmReachingEnv
from stable_baselines3 import SAC
import torch

print("\n" + "="*80)
print("GPU TRAINING TEST - FORCING CUDA")
print("="*80)

# Check GPU
print("\n[CHECK] GPU Status:")
os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# Force PyTorch to detect CUDA
print("\n[PYTORCH]")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")

# Try different device options
devices_to_try = ["cuda", "cuda:0", "cpu"]
for device_name in devices_to_try:
    try:
        test_tensor = torch.zeros(1).to(device_name)
        print(f"  ✓ Device '{device_name}' works!")
        best_device = device_name
        break
    except Exception as e:
        print(f"  ✗ Device '{device_name}' failed: {e}")
        best_device = "cpu"

# Create and train model
print(f"\n[MODEL] Creating SAC on device: {best_device}")
env = HabitatArmReachingEnv()

try:
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        device=best_device,
        verbose=1
    )
    print(f"✓ Model created on {best_device}")
    
    print("\n[TRAINING] Starting 5000 steps...")
    model.learn(total_timesteps=5000)
    print("✅ TRAINING SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

env.close()
print("\n" + "="*80 + "\n")
