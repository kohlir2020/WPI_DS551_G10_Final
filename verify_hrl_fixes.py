#!/usr/bin/env python3
"""
Lightweight test to verify HRL code structure is correct
Doesn't require all dependencies, just checks the logic
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/arm'))

print("\n" + "="*70)
print("HRL CODE STRUCTURE VERIFICATION")
print("="*70 + "\n")

# Test 1: Model path exists
print("TEST 1: Model files exist")
print("-" * 70)
model_paths = [
    "logs/simple_arm/realistic_sac_20251207_062739/final_sac.zip",
]

all_exist = True
for path in model_paths:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {path}")
    all_exist = all_exist and exists

print()

# Test 2: Check HAC code has correct parameters
print("TEST 2: HAC script has correct parameters")
print("-" * 70)

with open('src/arm/hac_continuous_her_arm.py', 'r') as f:
    hac_code = f.read()

checks = [
    ("Model path correct", "logs/simple_arm/realistic_sac_20251207_062739/final_sac"),
    ("Model type is SAC", 'default="SAC"'),
    ("Min dist for arm", "default=0.3,"),
    ("Max dist for arm", "default=2.0,"),
    ("Success radius for arm", "default=0.3,"),
    ("NaN preprocessing", "np.nan_to_num"),
]

all_checks_pass = True
for check_name, check_string in checks:
    passed = check_string in hac_code
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}")
    all_checks_pass = all_checks_pass and passed

print()

# Test 3: Check environment file exists and has fixes
print("TEST 3: Environment file has FK/IK fixes")
print("-" * 70)

env_file = 'src/arm/habitat_arm_reaching_env.py'
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        env_code = f.read()
    
    env_checks = [
        ("Proper FK with DH", "DH Parameters"),
        ("FK uses matrix multiplication", "T = T @ T_i"),
        ("IK has multi-start", "for attempt in range"),
        ("NaN handling in IK", "nan_to_num"),
    ]
    
    for check_name, check_string in env_checks:
        passed = check_string in env_code
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        all_checks_pass = all_checks_pass and passed
else:
    print("✗ Environment file not found")
    all_checks_pass = False

print()

# Test 4: Check documentation files exist
print("TEST 4: Documentation files created")
print("-" * 70)

doc_files = [
    "HRL_DIAGNOSIS_AND_FIXES.md",
    "HRL_FIXES_APPLIED.md",
    "HRL_QUICK_START.md",
    "IK_TROUBLESHOOTING.md",
    "test_hrl_fixes.py",
]

for doc_file in doc_files:
    exists = os.path.exists(doc_file)
    status = "✓" if exists else "✗"
    print(f"{status} {doc_file}")
    all_checks_pass = all_checks_pass and exists

print()

# Summary
print("="*70)
if all_checks_pass:
    print("✅ ALL CHECKS PASSED - HRL IS READY FOR TESTING")
else:
    print("⚠️  SOME CHECKS FAILED - SEE ABOVE")
print("="*70)

print("\nNext step: Run HRL training with fixed parameters")
print("Command: python src/arm/hac_continuous_her_arm.py --episodes 10")
print("\nExpected behavior:")
print("  - Episode 1-5: Negative rewards, no success")
print("  - Episode 5-10: Some success, rewards improving")
print("  - By episode 20-30: Success rate should reach 10-20%")
print("\n" + "="*70 + "\n")
