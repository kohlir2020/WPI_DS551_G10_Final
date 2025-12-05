"""
Compare RL Algorithm Results
Analyzes training curves and performance metrics for PPO, A2C, SAC
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_tensorboard_logs(log_dir):
    """Load training metrics from TensorBoard event files"""
    import subprocess
    
    # Convert event files to CSV
    result = subprocess.run(
        ["python", "-m", "tensorboard.plugins.export_csv", "--logdir", log_dir],
        capture_output=True,
        text=True
    )
    
    # Try to read CSV if it was created
    csv_files = list(Path(log_dir).glob("**/*.csv"))
    if csv_files:
        import pandas as pd
        return pd.read_csv(csv_files[0])
    return None


def analyze_algorithm(log_dir, algorithm_name):
    """Analyze training results for an algorithm"""
    
    # Find the main event file
    events_dir = Path(log_dir)
    if not events_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return None
    
    print(f"\n📊 {algorithm_name.upper()} Results")
    print(f"{'='*70}")
    print(f"Location: {log_dir}")
    
    # Count checkpoints
    checkpoint_dir = Path(log_dir) / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        print(f"✓ Checkpoints saved: {len(checkpoints)}")
        for ckpt in sorted(checkpoints)[-3:]:
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    
    # Check final model
    parent_dir = Path(log_dir)
    final_models = list(parent_dir.glob(f"final_{algorithm_name.lower()}.zip"))
    if final_models:
        size_mb = final_models[0].stat().st_size / (1024*1024)
        print(f"✓ Final model: {final_models[0].name} ({size_mb:.1f} MB)")
    
    return log_dir


def main():
    base_log_dir = "logs/simple_arm"
    
    # Find all algorithm directories
    ppo_dirs = sorted(Path(base_log_dir).glob("ppo_*"))
    a2c_dirs = sorted(Path(base_log_dir).glob("a2c_*"))
    sac_dirs = sorted(Path(base_log_dir).glob("sac_*"))
    
    print(f"\n{'='*70}")
    print(f"RL Algorithm Comparison - Training Results")
    print(f"{'='*70}")
    
    results = {}
    
    for log_dir in ppo_dirs:
        results["PPO"] = analyze_algorithm(log_dir, "PPO")
    
    for log_dir in a2c_dirs:
        results["A2C"] = analyze_algorithm(log_dir, "A2C")
    
    for log_dir in sac_dirs:
        results["SAC"] = analyze_algorithm(log_dir, "SAC")
    
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    
    if results:
        print(f"\n✅ Algorithms trained:")
        for algo in results:
            print(f"  - {algo}: {results[algo]}")
    
    print(f"\n📈 To visualize training progress:")
    print(f"   tensorboard --logdir {base_log_dir}")
    print(f"   Then open http://localhost:6006")
    
    print(f"\n📊 Next steps:")
    print(f"  1. Review learning curves in TensorBoard")
    print(f"  2. Compare rewards and stability")
    print(f"  3. Select best algorithm for vision-based training")
    print(f"  4. Implement multi-camera vision observations")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
