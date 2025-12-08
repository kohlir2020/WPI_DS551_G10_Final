#!/usr/bin/env python3
"""
Quick integration test - verify all components load correctly
"""
import os
import sys

sys.path.insert(0, 'src')

print("="*70)
print("TESTING INTEGRATION")
print("="*70)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from shared.scene_manager import create_fetch_scene
    from planner.llm_planner import TaskPlanner, get_hardcoded_plan
    from skill_executor import HRLNavigationSkill, ArmReachingSkill, execute_skill
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check model files exist
print("\n[2/5] Checking model files...")
models_to_check = [
    "src/navigation/models/lowlevel_curriculum_250k.zip",
    "src/navigation/models/hl_improved/highlevel_improved_final.zip",
    "logs/simple_arm/cartesian_ppo_20251207_011850/final_ppo.zip"
]
for model_path in models_to_check:
    if os.path.exists(model_path):
        print(f"✓ {model_path}")
    else:
        print(f"✗ Missing: {model_path}")

# Test 3: Test hardcoded planner
print("\n[3/5] Testing hardcoded planner...")
try:
    plan = get_hardcoded_plan("navigate_only")
    print(f"✓ Generated plan with {len(plan)} tasks")
    for i, task in enumerate(plan):
        print(f"  {i+1}. {task['skill']}: {task['params']}")
except Exception as e:
    print(f"✗ Planning failed: {e}")

# Test 4: Test LLM planner (if API key available)
print("\n[4/5] Testing LLM planner...")
if os.environ.get("OPENAI_API_KEY"):
    try:
        planner = TaskPlanner()
        plan = planner.plan_from_goal("navigate to the kitchen", [-16.0, 0.2, 5.0])
        if plan:
            print(f"✓ LLM generated plan with {len(plan)} tasks")
            for i, task in enumerate(plan):
                print(f"  {i+1}. {task['skill']}: {task['params']}")
        else:
            print("⚠️  LLM returned None (using fallback)")
    except Exception as e:
        print(f"⚠️  LLM planning error: {e}")
else:
    print("⚠️  OPENAI_API_KEY not set, skipping LLM test")

# Test 5: Available algorithms
print("\n[5/5] Available arm algorithms:")
arm_models = {
    "PPO": "logs/simple_arm/cartesian_ppo_20251207_011850/final_ppo.zip",
    "A2C": "logs/simple_arm/realistic_a2c_20251207_054618/final_a2c.zip",
    "SAC": "logs/simple_arm/realistic_sac_20251207_062739/final_sac.zip"
}
for algo, path in arm_models.items():
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"{exists} {algo}: {path}")

print("\n" + "="*70)
print("INTEGRATION TEST COMPLETE")
print("="*70)
print("\nTo run the system:")
print("  Without LLM: python src/main.py")
print("  With LLM:    python src/main.py --goal 'your goal here' --use-llm")
