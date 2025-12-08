#!/usr/bin/env python3
"""
Training Guide: Achieving 30-40% Success Rate with Video Recording

This script demonstrates the complete workflow to:
1. Train HRL to 30-40% success rate using improved algorithm
2. Record videos of successful episodes
3. Generate comprehensive reports and visualizations

Usage:
    python training_guide.py

Or follow manual steps below.
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime


class TrainingWorkflow:
    """Guide for complete HRL training and video generation workflow."""
    
    def __init__(self):
        self.workspace = Path("/home/adityapat/RL_final/WPI_DS551_G10_Final")
        self.logs_dir = self.workspace / "logs"
        self.models_dir = self.workspace / "hac_continuous_her_arm_models"
        self.reports_dir = self.workspace / "reports"
        self.videos_dir = self.workspace / "videos"
        
    def print_header(self, title: str):
        """Print section header."""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def print_step(self, step_num: int, title: str, description: str = ""):
        """Print step information."""
        print(f"\n[STEP {step_num}] {title}")
        if description:
            print(f"  → {description}")
    
    def show_overview(self):
        """Show overview of the workflow."""
        self.print_header("HRL Training Workflow: 30-40% Success Rate")
        
        print("OBJECTIVE:")
        print("  Improve HRL policy from 11% → 30-40% success rate")
        print("  Then record and visualize successful episodes\n")
        
        print("TIMELINE:")
        print("  Training:      ~45-60 minutes (for 300 episodes)")
        print("  Video Gen:     ~5-10 minutes (for 5 episodes)")
        print("  Report Gen:    ~2-3 minutes\n")
        
        print("OUTPUTS:")
        print("  ✓ Trained models (hl_actor.pth, hl_critic_*.pth)")
        print("  ✓ Training graphs (7 comprehensive reports)")
        print("  ✓ Success videos (MP4 format)")
        print("  ✓ Episode visualizations (trajectories, metrics)")
        
    def show_improvements(self):
        """Show what's improved since last version."""
        self.print_header("Algorithm Improvements")
        
        print("1. CURRICULUM LEARNING")
        print("   Before: Random goals anywhere (0-2m distance)")
        print("   After:  Progressive difficulty (0.5m → 2m over training)")
        print("   Impact: Easier early learning, smoother convergence\n")
        
        print("2. ENHANCED HINDSIGHT REPLAY (HER)")
        print("   Before: 4 future goals sampled per transition")
        print("   After:  8 future goals sampled per transition")
        print("   Impact: Better sample efficiency, faster learning\n")
        
        print("3. IMPROVED REWARD SHAPING")
        print("   Progress reward:  20 → 25  (better progress signal)")
        print("   Time penalty:     0.02 → 0.01  (encourage exploration)")
        print("   Success bonus:    50 → 100  (strongly reward reaching goal)")
        print("   Impact: Better learning signal, clearer success paths\n")
    
    def show_manual_steps(self):
        """Show manual training steps."""
        self.print_header("Manual Training Steps")
        
        self.print_step(1, "Train HRL with Curriculum Learning",
                       "Start training with improved algorithm and curriculum")
        print("""
        python src/arm/hac_continuous_her_arm.py \\
            --episodes 300 \\
            --use_curriculum \\
            --curriculum_start_dist 0.5 \\
            --curriculum_end_dist 2.0 \\
            --curriculum_episodes 150 \\
            --her_k_future 8 \\
            --log_interval 10
        
        Expected output:
          - Starts with easy goals (0.5m away) → closer success rate
          - Progressively increases goal distance over 150 episodes
          - Reaches full 2.0m distance around episode 150
          - Success rate should reach 30-40% by episode 300
          - Models saved to: hac_continuous_her_arm_models/TIMESTAMP_SAC/
        """)
        
        self.print_step(2, "Monitor Progress",
                       "Check success rates at key milestones")
        print("""
        Key indicators to watch:
          Episode 50:   ~5-10% success (learning on easy goals)
          Episode 100:  ~10-20% success (curriculum progressing)
          Episode 150:  ~15-25% success (moving to harder goals)
          Episode 200:  ~20-30% success (full difficulty)
          Episode 300:  ~30-40% success (target reached!)
        
        The success rate may dip around episode 150 (curriculum step).
        This is NORMAL - algorithm is adapting to harder goals.
        """)
        
        self.print_step(3, "Generate Training Reports",
                       "Create graphs of training progress")
        print("""
        python src/arm/generate_reports.py \\
            --log_dir logs \\
            --output_dir reports
        
        Generates 7 graphs:
          01_episode_rewards.png          - Reward curve with moving average
          02_success_rate.png             - Success rate progression
          03_final_distances.png          - Distribution of final distances
          04_loss_convergence.png         - Actor/critic loss curves
          05_subgoal_success.png          - Subgoal achievement rate
          06_exploration_noise.png        - Noise decay schedule
          07_combined_metrics.png         - All metrics in one dashboard
        """)
        
        self.print_step(4, "Record Success Videos",
                       "Record videos of learned policy")
        print("""
        python src/arm/record_fetch_video.py \\
            --model_path 'hac_continuous_her_arm_models/<YOUR_TIMESTAMP>/hl_actor.pth' \\
            --num_episodes 5 \\
            --output_dir videos
        
        Note: Replace <YOUR_TIMESTAMP> with actual directory
        
        Output:
          - episode_001.mp4 to episode_005.mp4
          - OR episode_001_SUCCESS.mp4 if successful
          - Shows 3D arm trajectory in Habitat environment
          - Real-time visualization of arm state and goal
        """)
        
        self.print_step(5, "Visualize Episode Trajectories",
                       "Create detailed episode analysis plots")
        print("""
        python src/arm/visualize_arm_episodes.py \\
            --log_dir logs/arm_episodes \\
            --output_dir episode_visualizations \\
            --num_episodes 10
        
        For each episode generates:
          - 2D trajectory (top-down view)
          - 3D trajectory with distance progression
          - Detailed metrics (rewards, actions, joint angles)
          - Success vs failure comparison
        """)
    
    def show_expected_results(self):
        """Show expected results."""
        self.print_header("Expected Results")
        
        print("TRAINING PERFORMANCE:")
        print("  ✓ Moving success rate: 11% → 30-40%")
        print("  ✓ Peak success episodes: Multiple episodes with 50-100+ reward")
        print("  ✓ Final distances: Many episodes ending < 0.3m from goal\n")
        
        print("VIDEOS:")
        print("  ✓ 3-5 successful episode videos recorded")
        print("  ✓ Shows arm smoothly reaching goals")
        print("  ✓ Real-time visualization of progress")
        print("  ✓ File size: ~100-500 KB per video\n")
        
        print("REPORTS:")
        print("  ✓ 7 comprehensive graphs generated")
        print("  ✓ Clear learning progression visible")
        print("  ✓ Convergence curves for actor/critic")
        print("  ✓ Success rate peaked at 30-40%\n")
    
    def show_troubleshooting(self):
        """Show troubleshooting guide."""
        self.print_header("Troubleshooting")
        
        print("ISSUE: Success rate not increasing (stuck at 5-10%)")
        print("  → Check that curriculum is enabled: --use_curriculum")
        print("  → Verify model path is correct")
        print("  → Try reducing learning rate: --hl_actor_lr 1e-3\n")
        
        print("ISSUE: Video recording fails")
        print("  → Check model path exists:")
        print("    ls hac_continuous_her_arm_models/*/hl_actor.pth")
        print("  → Install opencv-python: pip install opencv-python\n")
        
        print("ISSUE: Training crashes with memory error")
        print("  → Reduce batch size: --hl_batch 128")
        print("  → Reduce buffer size: --hl_buffer 50000\n")
        
        print("ISSUE: Reports show 'No episodes loaded'")
        print("  → Check episode logs exist: ls logs/episode_*.json")
        print("  → Training may not have completed yet\n")
    
    def show_commands_summary(self):
        """Show command summary for copy-paste."""
        self.print_header("Quick Command Reference")
        
        print("COMPLETE WORKFLOW (all at once):\n")
        print("# 1. Train with curriculum (45-60 min)")
        print("python src/arm/hac_continuous_her_arm.py --episodes 300 --use_curriculum\n")
        
        print("# 2. Find latest model directory")
        print("LATEST_MODEL=$(ls -td hac_continuous_her_arm_models/*/ | head -1)\n")
        
        print("# 3. Generate reports (2-3 min)")
        print("python src/arm/generate_reports.py --log_dir logs --output_dir reports\n")
        
        print("# 4. Record videos (5-10 min)")
        print("python src/arm/record_fetch_video.py \\")
        print("    --model_path $LATEST_MODEL/hl_actor.pth \\")
        print("    --num_episodes 5 --output_dir videos\n")
        
        print("# 5. Visualize episodes")
        print("python src/arm/visualize_arm_episodes.py \\")
        print("    --log_dir logs/arm_episodes \\")
        print("    --output_dir episode_viz\n")
    
    def run(self):
        """Run the complete guide."""
        self.show_overview()
        self.show_improvements()
        self.show_manual_steps()
        self.show_expected_results()
        self.show_troubleshooting()
        self.show_commands_summary()
        
        self.print_header("Ready to Train!")
        print("Start with Step 1 above to begin training for 30-40% success rate.")
        print("Monitor progress and follow remaining steps for videos and reports.")
        

if __name__ == "__main__":
    workflow = TrainingWorkflow()
    workflow.run()
