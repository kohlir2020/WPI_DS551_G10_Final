#!/usr/bin/env python3
"""
1M Step Training Script
Runs A2C and SAC for 1M steps each in sequence
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_training(algorithm, steps):
    """Run training for specified algorithm"""
    print("\n" + "="*70)
    print(f"Starting {algorithm} Training - {steps:,} steps")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cmd = [
        'python', 'src/arm/train_habitat.py',
        '--algorithm', algorithm,
        '--steps', str(steps),
        '--device', 'cuda'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd='/workspace')
    elapsed = time.time() - start_time
    
    hours = elapsed / 3600
    print(f"\n✅ {algorithm} completed in {hours:.1f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result.returncode == 0

def main():
    print("\n" + "="*70)
    print("1M STEP TRAINING SCHEDULE")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSequence:")
    print("  1. A2C: 1,000,000 steps (~5-6 hours)")
    print("  2. SAC: 1,000,000 steps (~5-6 hours)")
    print("\nTotal estimated time: ~10-12 hours")
    
    # Train A2C
    print("\n" + "="*70)
    print("PHASE 1: A2C Training")
    print("="*70)
    if not run_training('A2C', 1000000):
        print("❌ A2C training failed!")
        sys.exit(1)
    
    # Wait 2 minutes before SAC
    print("\n⏳ Waiting 2 minutes before starting SAC...")
    time.sleep(120)
    
    # Train SAC
    print("\n" + "="*70)
    print("PHASE 2: SAC Training")
    print("="*70)
    if not run_training('SAC', 1000000):
        print("❌ SAC training failed!")
        sys.exit(1)
    
    # Done
    print("\n" + "="*70)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Compare results: python3 compare_all_phases.py --final")
    print("  2. Push to git: git add . && git commit -m '1M training results' && git push")

if __name__ == '__main__':
    main()
