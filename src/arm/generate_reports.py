#!/usr/bin/env python3
"""
Generate comprehensive performance reports and graphs for HRL training.

This script loads training logs and generates:
1. Episode reward curves (with smoothing)
2. Success rate curves (moving average)
3. Final distance distributions
4. Actor and critic loss convergence
5. Subgoal achievement rates
6. Exploration noise decay
7. Combined convergence plot
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Tuple, Optional
import argparse


class TrainingReportGenerator:
    """Generate training reports from logged metrics."""
    
    def __init__(self, log_dir: str, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            log_dir: Directory containing training logs (episode_*.json files)
            output_dir: Directory to save generated reports
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all episode logs
        self.episodes = self._load_episodes()
        
    def _load_episodes(self) -> Dict:
        """Load all episode data from log files."""
        episodes = {}
        
        log_files = sorted(self.log_dir.glob("episode_*.json"))
        print(f"Found {len(log_files)} episode log files")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    ep_num = int(log_file.stem.split('_')[1])
                    episodes[ep_num] = data
            except Exception as e:
                print(f"Warning: Failed to load {log_file}: {e}")
        
        if not episodes:
            print(f"WARNING: No episode logs found in {self.log_dir}")
            return {}
        
        print(f"Loaded {len(episodes)} episodes successfully")
        return episodes
    
    def _extract_metric(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract metric values across all episodes.
        
        Args:
            key: Key name in episode data (e.g., 'reward', 'success', 'final_dist')
            
        Returns:
            (episode_numbers, values) tuple
        """
        episodes = sorted(self.episodes.keys())
        values = []
        
        for ep in episodes:
            if key in self.episodes[ep]:
                values.append(self.episodes[ep][key])
            else:
                values.append(np.nan)
        
        return np.array(episodes), np.array(values)
    
    def _smooth_curve(self, values: np.ndarray, window: int = 10) -> np.ndarray:
        """Smooth curve using moving average."""
        if len(values) < window:
            return values
        return uniform_filter1d(values, size=window, mode='nearest')
    
    def plot_episode_rewards(self, window: int = 10, figsize: Tuple = (12, 6)):
        """
        Plot episode rewards with moving average.
        
        Args:
            window: Size of moving average window
            figsize: Figure size
        """
        if not self.episodes:
            print("No episodes loaded, skipping reward plot")
            return
        
        episodes, rewards = self._extract_metric('reward')
        
        if len(rewards) == 0 or np.all(np.isnan(rewards)):
            print("No reward data found")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Raw rewards
        ax.plot(episodes, rewards, 'o-', alpha=0.3, label='Episode Reward', markersize=4)
        
        # Smoothed rewards
        smoothed = self._smooth_curve(rewards, window)
        ax.plot(episodes, smoothed, 'b-', linewidth=2, label=f'Moving Avg (window={window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / "01_episode_rewards.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_success_rate(self, window: int = 20, figsize: Tuple = (12, 6)):
        """
        Plot success rate with moving average.
        
        Args:
            window: Size of moving average window
            figsize: Figure size
        """
        if not self.episodes:
            print("No episodes loaded, skipping success rate plot")
            return
        
        episodes, successes = self._extract_metric('success')
        
        if len(successes) == 0 or np.all(np.isnan(successes)):
            print("No success data found")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate moving success rate
        moving_success = []
        for i in range(len(successes)):
            start = max(0, i - window + 1)
            window_data = successes[start:i+1]
            moving_success.append(np.nanmean(window_data) * 100)
        
        moving_success = np.array(moving_success)
        
        # Plot
        ax.plot(episodes, moving_success, 'g-', linewidth=2.5, label=f'Success Rate (window={window})')
        ax.fill_between(episodes, moving_success, alpha=0.2, color='g')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Moving Success Rate', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / "02_success_rate.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_final_distances(self, figsize: Tuple = (12, 6)):
        """Plot final distance to goal distribution."""
        if not self.episodes:
            print("No episodes loaded, skipping final distances plot")
            return
        
        episodes, distances = self._extract_metric('final_dist')
        
        if len(distances) == 0 or np.all(np.isnan(distances)):
            print("No final distance data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        ax1.scatter(episodes, distances, alpha=0.5, s=30, color='purple')
        ax1.axhline(y=0.3, color='r', linestyle='--', linewidth=2, label='Success Radius (0.3m)')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Final Distance to Goal (m)', fontsize=12)
        ax1.set_title('Final Distance Per Episode', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Histogram
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) > 0:
            ax2.hist(valid_distances, bins=30, color='purple', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0.3, color='r', linestyle='--', linewidth=2, label='Success Threshold')
            ax2.set_xlabel('Distance to Goal (m)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Distance Distribution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "03_final_distances.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_loss_convergence(self, window: int = 10, figsize: Tuple = (14, 8)):
        """Plot actor and critic loss convergence."""
        if not self.episodes:
            print("No episodes loaded, skipping loss convergence plot")
            return
        
        episodes, critic_losses = self._extract_metric('critic_loss')
        _, actor_losses = self._extract_metric('actor_loss')
        
        if len(critic_losses) == 0 and len(actor_losses) == 0:
            print("No loss data found - may need to run training with updated code")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Critic loss
        if not np.all(np.isnan(critic_losses)):
            ax = axes[0, 0]
            valid_mask = ~np.isnan(critic_losses)
            valid_episodes = episodes[valid_mask]
            valid_losses = critic_losses[valid_mask]
            
            ax.plot(valid_episodes, valid_losses, 'o-', alpha=0.3, markersize=4, color='red')
            smoothed = self._smooth_curve(valid_losses, window)
            ax.plot(valid_episodes, smoothed, 'r-', linewidth=2.5, label=f'Smoothed (w={window})')
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Critic Loss', fontsize=11)
            ax.set_title('Critic Loss Convergence', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        # Actor loss
        if not np.all(np.isnan(actor_losses)):
            ax = axes[0, 1]
            valid_mask = ~np.isnan(actor_losses)
            valid_episodes = episodes[valid_mask]
            valid_losses = actor_losses[valid_mask]
            
            ax.plot(valid_episodes, valid_losses, 'o-', alpha=0.3, markersize=4, color='orange')
            smoothed = self._smooth_curve(valid_losses, window)
            ax.plot(valid_episodes, smoothed, 'orange', linewidth=2.5, label=f'Smoothed (w={window})')
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Actor Loss', fontsize=11)
            ax.set_title('Actor Loss Convergence', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        # Loss histograms
        ax = axes[1, 0]
        valid_critic = critic_losses[~np.isnan(critic_losses)]
        if len(valid_critic) > 0:
            ax.hist(valid_critic, bins=30, alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('Critic Loss Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Critic Loss Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1, 1]
        valid_actor = actor_losses[~np.isnan(actor_losses)]
        if len(valid_actor) > 0:
            ax.hist(valid_actor, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Actor Loss Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Actor Loss Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "04_loss_convergence.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_subgoal_success(self, figsize: Tuple = (12, 6)):
        """Plot subgoal achievement rate."""
        if not self.episodes:
            print("No episodes loaded, skipping subgoal plot")
            return
        
        episodes, subgoals = self._extract_metric('subgoal_successes')
        
        if len(subgoals) == 0 or np.all(np.isnan(subgoals)):
            print("No subgoal data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subgoals achieved per episode
        valid_mask = ~np.isnan(subgoals)
        ax1.bar(episodes[valid_mask], subgoals[valid_mask], color='teal', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Subgoals Achieved', fontsize=12)
        ax1.set_title('Subgoal Achievements Per Episode', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Average subgoals with trend
        window = 20
        moving_avg = []
        for i in range(len(subgoals)):
            start = max(0, i - window + 1)
            valid_window = subgoals[start:i+1]
            valid_window = valid_window[~np.isnan(valid_window)]
            if len(valid_window) > 0:
                moving_avg.append(np.mean(valid_window))
            else:
                moving_avg.append(np.nan)
        
        moving_avg = np.array(moving_avg)
        ax2.plot(episodes, moving_avg, 'teal', linewidth=2.5, label=f'Moving Avg (w={window})')
        ax2.fill_between(episodes, moving_avg, alpha=0.2, color='teal')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Avg Subgoals Achieved', fontsize=12)
        ax2.set_title('Moving Average Subgoal Achievement', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / "05_subgoal_success.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_exploration_noise(self, figsize: Tuple = (12, 6)):
        """Plot exploration noise decay."""
        if not self.episodes:
            print("No episodes loaded, skipping exploration noise plot")
            return
        
        episodes, noise = self._extract_metric('exploration_noise')
        
        if len(noise) == 0 or np.all(np.isnan(noise)):
            print("No exploration noise data found")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        valid_mask = ~np.isnan(noise)
        valid_episodes = episodes[valid_mask]
        valid_noise = noise[valid_mask]
        
        ax.plot(valid_episodes, valid_noise, 'o-', alpha=0.5, markersize=5, color='navy')
        
        # Add theoretical decay curve
        # Assuming linear decay from initial to zero over hl_noise_decay_episodes=500
        if len(valid_episodes) > 0:
            initial_noise = valid_noise[0]
            decay_episodes = 500
            theoretical = np.maximum(0, initial_noise * (1 - valid_episodes / decay_episodes))
            ax.plot(valid_episodes, theoretical, '--', color='red', linewidth=2, 
                   label='Theoretical Decay (500-episode)')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Exploration Noise Std', fontsize=12)
        ax.set_title('Exploration Noise Decay Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / "06_exploration_noise.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_combined_metrics(self, figsize: Tuple = (16, 10)):
        """Create combined multi-panel figure with key metrics."""
        if not self.episodes:
            print("No episodes loaded, skipping combined plot")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        episodes, rewards = self._extract_metric('reward')
        _, successes = self._extract_metric('success')
        _, distances = self._extract_metric('final_dist')
        _, critic_losses = self._extract_metric('critic_loss')
        _, actor_losses = self._extract_metric('actor_loss')
        _, subgoals = self._extract_metric('subgoal_successes')
        _, noise = self._extract_metric('exploration_noise')
        
        # 1. Rewards
        ax = fig.add_subplot(gs[0, 0])
        if len(rewards) > 0 and not np.all(np.isnan(rewards)):
            smoothed = self._smooth_curve(rewards, 10)
            ax.plot(episodes, smoothed, 'b-', linewidth=2)
            ax.fill_between(episodes, smoothed, alpha=0.2, color='b')
            ax.set_title('Episode Rewards', fontweight='bold')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # 2. Success Rate
        ax = fig.add_subplot(gs[0, 1])
        if len(successes) > 0 and not np.all(np.isnan(successes)):
            moving_success = []
            for i in range(len(successes)):
                start = max(0, i - 19)
                window_data = successes[start:i+1]
                moving_success.append(np.nanmean(window_data) * 100)
            ax.plot(episodes, moving_success, 'g-', linewidth=2)
            ax.fill_between(episodes, moving_success, alpha=0.2, color='g')
            ax.set_title('Success Rate (20-ep window)', fontweight='bold')
            ax.set_ylabel('Success %')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3)
        
        # 3. Final Distances
        ax = fig.add_subplot(gs[0, 2])
        if len(distances) > 0 and not np.all(np.isnan(distances)):
            ax.scatter(episodes, distances, alpha=0.5, s=20, color='purple')
            ax.axhline(y=0.3, color='r', linestyle='--', linewidth=2)
            ax.set_title('Final Distance to Goal', fontweight='bold')
            ax.set_ylabel('Distance (m)')
            ax.grid(True, alpha=0.3)
        
        # 4. Critic Loss
        ax = fig.add_subplot(gs[1, 0])
        if len(critic_losses) > 0 and not np.all(np.isnan(critic_losses)):
            valid_mask = ~np.isnan(critic_losses)
            valid_eps = episodes[valid_mask]
            valid_loss = critic_losses[valid_mask]
            smoothed = self._smooth_curve(valid_loss, 10)
            ax.plot(valid_eps, smoothed, 'r-', linewidth=2)
            ax.set_title('Critic Loss', fontweight='bold')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # 5. Actor Loss
        ax = fig.add_subplot(gs[1, 1])
        if len(actor_losses) > 0 and not np.all(np.isnan(actor_losses)):
            valid_mask = ~np.isnan(actor_losses)
            valid_eps = episodes[valid_mask]
            valid_loss = actor_losses[valid_mask]
            smoothed = self._smooth_curve(valid_loss, 10)
            ax.plot(valid_eps, smoothed, 'orange', linewidth=2)
            ax.set_title('Actor Loss', fontweight='bold')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # 6. Subgoals
        ax = fig.add_subplot(gs[1, 2])
        if len(subgoals) > 0 and not np.all(np.isnan(subgoals)):
            valid_mask = ~np.isnan(subgoals)
            ax.bar(episodes[valid_mask], subgoals[valid_mask], color='teal', alpha=0.7, width=0.8)
            ax.set_title('Subgoals Achieved', fontweight='bold')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 7. Noise
        ax = fig.add_subplot(gs[2, 0])
        if len(noise) > 0 and not np.all(np.isnan(noise)):
            valid_mask = ~np.isnan(noise)
            ax.plot(episodes[valid_mask], noise[valid_mask], 'navy', linewidth=2)
            ax.set_title('Exploration Noise', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Noise Std')
            ax.grid(True, alpha=0.3)
        
        # 8. Reward Distribution
        ax = fig.add_subplot(gs[2, 1])
        if len(rewards) > 0 and not np.all(np.isnan(rewards)):
            valid_rewards = rewards[~np.isnan(rewards)]
            ax.hist(valid_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title('Reward Distribution', fontweight='bold')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 9. Statistics Table
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        # Compute statistics
        stats_text = "Training Statistics:\n\n"
        
        if len(rewards) > 0 and not np.all(np.isnan(rewards)):
            valid_rewards = rewards[~np.isnan(rewards)]
            stats_text += f"Rewards:\n"
            stats_text += f"  Mean: {np.mean(valid_rewards):.2f}\n"
            stats_text += f"  Max: {np.max(valid_rewards):.2f}\n"
            stats_text += f"  Min: {np.min(valid_rewards):.2f}\n\n"
        
        if len(successes) > 0 and not np.all(np.isnan(successes)):
            success_rate = np.nanmean(successes) * 100
            stats_text += f"Overall Success: {success_rate:.1f}%\n\n"
        
        if len(distances) > 0 and not np.all(np.isnan(distances)):
            valid_dist = distances[~np.isnan(distances)]
            success_dist = np.sum(valid_dist < 0.3)
            stats_text += f"Successful Episodes: {success_dist}/{len(valid_dist)}\n"
            stats_text += f"Mean Distance: {np.mean(valid_dist):.3f}m\n\n"
        
        stats_text += f"Total Episodes: {len(episodes)}"
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('HRL Training: Combined Performance Metrics', fontsize=16, fontweight='bold')
        
        output_path = self.output_dir / "07_combined_metrics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_reports(self):
        """Generate all available reports."""
        print("\n" + "="*60)
        print("Generating comprehensive training reports...")
        print("="*60 + "\n")
        
        if not self.episodes:
            print("ERROR: No episodes loaded. Make sure log files exist.")
            return
        
        print(f"Generating reports for {len(self.episodes)} episodes...\n")
        
        self.plot_episode_rewards()
        self.plot_success_rate()
        self.plot_final_distances()
        self.plot_loss_convergence()
        self.plot_subgoal_success()
        self.plot_exploration_noise()
        self.plot_combined_metrics()
        
        print("\n" + "="*60)
        print(f"✓ All reports saved to: {self.output_dir}")
        print("="*60)
        
        # List all generated files
        generated_files = sorted(self.output_dir.glob("*.png"))
        print(f"\nGenerated {len(generated_files)} graph files:")
        for f in generated_files:
            print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate HRL training reports")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory containing episode logs")
    parser.add_argument("--output_dir", type=str, default="reports",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    generator = TrainingReportGenerator(args.log_dir, args.output_dir)
    generator.generate_all_reports()


if __name__ == "__main__":
    main()
