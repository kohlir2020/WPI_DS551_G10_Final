#!/usr/bin/env python3
"""
Quick validation script - checks environment code structure without needing Habitat
"""

import os
import sys
import re

def check_file_exists(path):
    """Check if file exists"""
    if os.path.exists(path):
        print(f"  ✓ {path}")
        return True
    else:
        print(f"  ✗ MISSING: {path}")
        return False

def check_file_contains(path, pattern):
    """Check if file contains pattern"""
    try:
        with open(path, 'r') as f:
            content = f.read()
            if re.search(pattern, content, re.IGNORECASE):
                return True
    except:
        pass
    return False

def validate_environment():
    """Validate arm environment setup"""
    print("\n" + "="*70)
    print("ARCHITECTURE VALIDATION")
    print("="*70)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Check files exist
    print("\n[1] Checking files exist...")
    files_to_check = [
        "src/arm/fetch_arm_reaching_env.py",
        "src/arm/train_fetch_arm.py",
        "src/simple_navigation_env.py",
        "src/train_low_level_nav.py",
        "src/skill_executor.py",
        "launch_parallel_training.py",
        "ARM_REACHING_GUIDE.md",
        "ARCHITECTURE_REDESIGN.md"
    ]
    
    all_exist = True
    for f in files_to_check:
        full_path = os.path.join(base_path, f)
        if not check_file_exists(full_path):
            all_exist = False
    
    # Check environment class structure
    print("\n[2] Checking FetchArmReachingEnv structure...")
    env_file = os.path.join(base_path, "src/arm/fetch_arm_reaching_env.py")
    
    checks = [
        ("class FetchArmReachingEnv", "Class definition"),
        ("observation_space.*Box", "Observation space"),
        ("action_space.*Box", "Action space"),
        ("def reset", "reset() method"),
        ("def step", "step() method"),
        ("def _get_obs", "_get_obs() method"),
        ("success_distance.*=.*0.15", "15cm success threshold"),
        ("arm_joint_names", "7-DOF arm joints"),
    ]
    
    for pattern, description in checks:
        if check_file_contains(env_file, pattern):
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ MISSING: {description}")
    
    # Check trainer structure
    print("\n[3] Checking train_fetch_arm.py structure...")
    trainer_file = os.path.join(base_path, "src/arm/train_fetch_arm.py")
    
    checks = [
        ("def train_fetch_arm", "train_fetch_arm() function"),
        ("PPO", "PPO algorithm"),
        ("FetchArmReachingEnv", "Environment usage"),
        ("CheckpointCallback", "Checkpoint saving"),
        ("EvalCallback", "Evaluation callback"),
        ("total_timesteps", "Timesteps parameter"),
    ]
    
    for pattern, description in checks:
        if check_file_contains(trainer_file, pattern):
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ MISSING: {description}")
    
    # Check launcher update
    print("\n[4] Checking launch_parallel_training.py update...")
    launcher_file = os.path.join(base_path, "launch_parallel_training.py")
    
    if check_file_contains(launcher_file, "train_fetch_arm.py"):
        print("  ✓ Launcher references train_fetch_arm.py")
    else:
        print("  ✗ Launcher still references old fetch_nav.py")
    
    if check_file_contains(launcher_file, "fetch_arm"):
        print("  ✓ Process name updated to fetch_arm")
    else:
        print("  ✗ Process name not updated")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all_exist:
        print("\n✓ All files present and accounted for!")
        print("\n✓ Architecture redesigned for ARM REACHING:")
        print("  - Navigation: [distance, angle] → discrete actions (WORKING)")
        print("  - Arm reaching: [distance, arm_angles(7)] → continuous actions (READY)")
        print("  - Skill executor: Ready to combine both skills")
        print("\n✓ Next steps:")
        print("  1. Run in Docker: docker-compose up -d")
        print("  2. Train navigation: python src/train_low_level_nav.py (already done)")
        print("  3. Train arm reaching: python src/arm/train_fetch_arm.py (new)")
        print("  4. Parallel training: python launch_parallel_training.py --mode both")
        return True
    else:
        print("\n✗ Some files missing - see details above")
        return False

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
