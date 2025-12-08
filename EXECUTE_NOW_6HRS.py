#!/usr/bin/env python3
"""
CRITICAL: 6-HOUR EXECUTION PLAN FOR HRL 30-40% SUCCESS + VIDEO

Deadline: 6 hours from now
Task: Train HRL to 30-40% success, record video of successful episodes

=== PHASE 1: QUICK TRAINING (45-60 minutes) ===
Goal: Achieve 30-40% success rate with curriculum learning

Commands to run (in order):
"""

import subprocess
import sys
from pathlib import Path

PHASE_1 = """
# Phase 1: Train HRL with curriculum learning for 30-40% success
cd /home/adityapat/RL_final/WPI_DS551_G10_Final

# CRITICAL: RUN THIS TRAINING COMMAND NOW
python3 src/arm/hac_continuous_her_arm.py \\
    --episodes 300 \\
    --use_curriculum \\
    --curriculum_start_dist 0.5 \\
    --curriculum_end_dist 2.0 \\
    --curriculum_episodes 150 \\
    --her_k_future 8 \\
    --hl_progress_scale 25.0 \\
    --hl_time_penalty 0.01 \\
    --hl_success_bonus 100.0 \\
    --log_interval 10

# Expected: 45-60 minutes
# Expected success rate: 30-40% by episode 300
# Output: logs/ directory with training data
"""

PHASE_2 = """
=== PHASE 2: MONITOR & REPORT (10 minutes) ===

# Generate comprehensive training graphs
python3 src/arm/generate_reports.py --log_dir logs --output_dir reports

# Check success rate
ls reports/*.png
# Should have 7 graphs showing success rate progression

# Expected: Success rate curve showing 30-40% final rate
"""

PHASE_3 = """
=== PHASE 3: GENERATE VIDEOS (15 minutes) ===

# Record videos of successful episodes
python3 src/arm/record_fetch_video.py \\
    --num_episodes 3 \\
    --output_dir videos/final_demo

# Expected: 3 MP4 videos (some successful, some not)
# Total time: ~5 minutes for 3 episodes
"""

PHASE_4 = """
=== PHASE 4: VISUALIZE TRAJECTORIES (10 minutes) ===

# Create detailed trajectory visualizations
python3 src/arm/visualize_arm_episodes.py \\
    --log_dir test_episode_logs \\
    --output_dir visualizations \\
    --num_episodes 5

# Expected: Multiple PNG files showing 3D trajectories
"""

PHASE_5 = """
=== PHASE 5: FINAL DELIVERABLES (20 minutes) ===

# Collect all outputs
mkdir final_results
cp -r reports/ final_results/
cp -r videos/ final_results/
cp -r visualizations/ final_results/
cp src/arm/*.py final_results/

# Create summary
echo "=== PROJECT SUMMARY ===" > final_results/SUMMARY.txt
echo "Success Rate: Check reports/07_combined_metrics.png" >> final_results/SUMMARY.txt
echo "Videos: Watch videos/final_demo/*.mp4" >> final_results/SUMMARY.txt
echo "Visualizations: See visualizations/ folder" >> final_results/SUMMARY.txt

# Push everything to git
git add -A
git commit -m "Final deliverables: 30-40% success rate with video demonstrations"
git push origin HEAD:aditya/arm
"""

TIMELINE = """
=== 6-HOUR TIMELINE ===

00:00 - 01:00  | PHASE 1: Run training (--use_curriculum for 30-40%)
01:00 - 01:15  | PHASE 2: Generate training reports (7 graphs)
01:15 - 01:30  | PHASE 3: Record 3 videos of policy performance
01:30 - 01:45  | PHASE 4: Visualize episode trajectories
01:45 - 02:05  | PHASE 5: Compile final deliverables
02:05 - 02:15  | Buffer: Handle any issues
02:15 - 06:00  | Report writing & final assembly (if needed)

CRITICAL SUCCESS FACTORS:
✓ Use --use_curriculum flag (NOT optional - needed for 30-40%)
✓ Run for 300 episodes (target: episode 250-300 shows 30-40%)
✓ Save videos before closing terminal
✓ Keep reports folder for metrics evidence
✓ Push everything to git for backup
"""

QUICK_REFERENCE = """
=== QUICK COMMAND REFERENCE ===

# IMMEDIATE ACTION: Start training NOW
cd /home/adityapat/RL_final/WPI_DS551_G10_Final && python3 src/arm/hac_continuous_her_arm.py --episodes 300 --use_curriculum

# After training completes:
python3 src/arm/generate_reports.py --log_dir logs --output_dir reports

# Record videos:
python3 src/arm/record_fetch_video.py --num_episodes 3 --output_dir videos/final

# Final push:
git add -A && git commit -m "Final project: 30-40% HRL success rate" && git push
"""

if __name__ == "__main__":
    print(PHASE_1)
    print("\n")
    print(PHASE_2)
    print("\n")
    print(PHASE_3)
    print("\n")
    print(PHASE_4)
    print("\n")
    print(PHASE_5)
    print("\n")
    print(TIMELINE)
    print("\n")
    print(QUICK_REFERENCE)
    
    print("\n" + "="*70)
    print("CRITICAL: Start Phase 1 training immediately!")
    print("="*70)
    print("\nRun this NOW:")
    print("python3 src/arm/hac_continuous_her_arm.py --episodes 300 --use_curriculum")
