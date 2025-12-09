#!/usr/bin/env python3
"""
Visualization script for HRL arm reaching task episodes.

This script loads trained models and generates visualizations of:
1. Arm trajectory visualizations (3D arm poses)
2. Episode progression plots (distance, reward over steps)
3. Success vs failure comparisons
4. Policy rollout demonstrations

Usage:
    python visualize_arm_episodes.py --model_path logs/model.pt --num_episodes 5
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'habitat-lab'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from arm_reaching_env import HabitatArmReachingEnv
except ImportError:
    print("Warning: Could not import HabitatArmReachingEnv directly")


class ArmEpisodeVisualizer:
    """Visualize arm reaching episodes."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_episode_trajectory(self, episode_file: str) -> Optional[Dict]:
        """
        Load episode trajectory from saved file.
        
        Args:
            episode_file: Path to episode JSON/pickle file
            
        Returns:
            Episode data dictionary with trajectory, rewards, etc.
        """
        try:
            if episode_file.endswith('.json'):
                with open(episode_file, 'r') as f:
                    return json.load(f)
            elif episode_file.endswith('.pkl'):
                with open(episode_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load episode {episode_file}: {e}")
        return None
    
    def plot_2d_trajectory(self, episode_data: Dict, title: str = "Arm Trajectory",
                          output_name: str = "trajectory_2d"):
        """
        Plot 2D arm trajectory (top-down view).
        
        Args:
            episode_data: Episode trajectory data
            title: Plot title
            output_name: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract trajectory if available
        if 'ee_positions' in episode_data:
            ee_pos = np.array(episode_data['ee_positions'])
            
            # XY plane view
            ax = axes[0]
            ax.plot(ee_pos[:, 0], ee_pos[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.plot(ee_pos[0, 0], ee_pos[0, 1], 'go', markersize=12, label='Start')
            ax.plot(ee_pos[-1, 0], ee_pos[-1, 1], 'r*', markersize=20, label='End')
            
            if 'goal_position' in episode_data:
                goal = episode_data['goal_position']
                circle = patches.Circle((goal[0], goal[1]), 0.3, 
                                       fill=False, edgecolor='red', linestyle='--', linewidth=2)
                ax.add_patch(circle)
                ax.plot(goal[0], goal[1], 'r+', markersize=15, markeredgewidth=2)
            
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_title('Top-Down View (XY Plane)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.axis('equal')
            
            # XZ plane view
            ax = axes[1]
            ax.plot(ee_pos[:, 0], ee_pos[:, 2], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.plot(ee_pos[0, 0], ee_pos[0, 2], 'go', markersize=12, label='Start')
            ax.plot(ee_pos[-1, 0], ee_pos[-1, 2], 'r*', markersize=20, label='End')
            
            if 'goal_position' in episode_data:
                goal = episode_data['goal_position']
                circle = patches.Circle((goal[0], goal[2]), 0.3,
                                       fill=False, edgecolor='red', linestyle='--', linewidth=2)
                ax.add_patch(circle)
                ax.plot(goal[0], goal[2], 'r+', markersize=15, markeredgewidth=2)
            
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Z (m)', fontsize=11)
            ax.set_title('Front View (XZ Plane)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.axis('equal')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_3d_trajectory(self, episode_data: Dict, title: str = "3D Arm Trajectory",
                          output_name: str = "trajectory_3d"):
        """
        Plot 3D arm trajectory.
        
        Args:
            episode_data: Episode trajectory data
            title: Plot title
            output_name: Output filename
        """
        fig = plt.figure(figsize=(14, 6))
        
        if 'ee_positions' in episode_data:
            ee_pos = np.array(episode_data['ee_positions'])
            
            # 3D trajectory
            ax = fig.add_subplot(121, projection='3d')
            ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], 'b-', linewidth=2, alpha=0.7)
            ax.scatter(*ee_pos[0], color='green', s=100, marker='o', label='Start')
            ax.scatter(*ee_pos[-1], color='red', s=200, marker='*', label='End')
            
            if 'goal_position' in episode_data:
                goal = episode_data['goal_position']
                ax.scatter(*goal, color='red', s=150, marker='+', label='Goal')
                
                # Draw sphere around goal
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = goal[0] + 0.3 * np.outer(np.cos(u), np.sin(v))
                y = goal[1] + 0.3 * np.outer(np.sin(u), np.sin(v))
                z = goal[2] + 0.3 * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.1, color='red')
            
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_zlabel('Z (m)', fontsize=11)
            ax.set_title('3D End-Effector Trajectory', fontweight='bold')
            ax.legend(fontsize=10)
            
            # Distance progression
            ax = fig.add_subplot(122)
            distances = []
            if 'distances_to_goal' in episode_data:
                distances = episode_data['distances_to_goal']
            else:
                # Calculate from positions
                goal = np.array(episode_data['goal_position'])
                distances = [np.linalg.norm(pos - goal) for pos in ee_pos]
            
            steps = np.arange(len(distances))
            ax.plot(steps, distances, 'g-', linewidth=2, label='Distance to Goal')
            ax.axhline(y=0.3, color='r', linestyle='--', linewidth=2, label='Success Threshold')
            ax.fill_between(steps, distances, alpha=0.2, color='g')
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Distance (m)', fontsize=11)
            ax.set_title('Distance Progression', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_episode_metrics(self, episode_data: Dict, title: str = "Episode Metrics",
                            output_name: str = "episode_metrics"):
        """
        Plot episode-level metrics (reward, actions, joint angles).
        
        Args:
            episode_data: Episode data
            title: Plot title
            output_name: Output filename
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # 1. Cumulative reward
        ax = fig.add_subplot(gs[0, 0])
        if 'rewards' in episode_data:
            rewards = episode_data['rewards']
            steps = np.arange(len(rewards))
            cum_reward = np.cumsum(rewards)
            ax.plot(steps, cum_reward, 'b-', linewidth=2)
            ax.fill_between(steps, cum_reward, alpha=0.2, color='b')
            ax.set_title('Cumulative Reward', fontweight='bold')
            ax.set_ylabel('Cumulative Reward')
            ax.grid(True, alpha=0.3)
        
        # 2. Step rewards
        ax = fig.add_subplot(gs[0, 1])
        if 'rewards' in episode_data:
            rewards = episode_data['rewards']
            steps = np.arange(len(rewards))
            ax.bar(steps, rewards, color='orange', alpha=0.7, edgecolor='black', width=0.8)
            ax.set_title('Step Rewards', fontweight='bold')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Distance progression
        ax = fig.add_subplot(gs[0, 2])
        if 'distances_to_goal' in episode_data:
            distances = episode_data['distances_to_goal']
            steps = np.arange(len(distances))
            ax.plot(steps, distances, 'g-', linewidth=2)
            ax.axhline(y=0.3, color='r', linestyle='--', linewidth=2)
            ax.fill_between(steps, distances, alpha=0.2, color='g')
            ax.set_title('Distance to Goal', fontweight='bold')
            ax.set_ylabel('Distance (m)')
            ax.grid(True, alpha=0.3)
        
        # 4-6. Action components
        if 'actions' in episode_data:
            actions = np.array(episode_data['actions'])
            steps = np.arange(len(actions))
            
            for idx, (ax_idx, axis_name) in enumerate([(gs[1, 0], 'X'), 
                                                        (gs[1, 1], 'Y'), 
                                                        (gs[1, 2], 'Z')]):
                ax = fig.add_subplot(ax_idx)
                if idx < actions.shape[1]:
                    ax.plot(steps, actions[:, idx], 'purple', linewidth=1.5, alpha=0.7)
                    ax.fill_between(steps, actions[:, idx], alpha=0.2, color='purple')
                    ax.set_title(f'Action {axis_name}', fontweight='bold')
                    ax.set_ylabel('Action Value')
                    ax.grid(True, alpha=0.3)
        
        # 7-9. Joint angles
        if 'joint_angles' in episode_data:
            joint_angles = np.array(episode_data['joint_angles'])
            steps = np.arange(len(joint_angles))
            
            # Pick 3 joints to display
            for idx, joint_idx in enumerate([0, 3, 6]):
                ax = fig.add_subplot(GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)[2, idx])
                if joint_idx < joint_angles.shape[1]:
                    ax.plot(steps, joint_angles[:, joint_idx], 'teal', linewidth=1.5, alpha=0.7)
                    ax.fill_between(steps, joint_angles[:, joint_idx], alpha=0.2, color='teal')
                    ax.set_title(f'Joint {joint_idx}', fontweight='bold')
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Angle (rad)')
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_success_comparison(self, successful_episodes: List[Dict],
                               failed_episodes: List[Dict],
                               output_name: str = "success_comparison"):
        """
        Compare successful vs failed episodes.
        
        Args:
            successful_episodes: List of successful episode data
            failed_episodes: List of failed episode data
            output_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Average trajectory comparison
        ax = axes[0, 0]
        if successful_episodes and 'distances_to_goal' in successful_episodes[0]:
            success_dists = [np.array(ep['distances_to_goal']) for ep in successful_episodes]
            fail_dists = [np.array(ep['distances_to_goal']) for ep in failed_episodes]
            
            max_len = max(max(len(d) for d in success_dists) if success_dists else 0,
                         max(len(d) for d in fail_dists) if fail_dists else 0)
            
            # Pad trajectories
            success_padded = []
            for d in success_dists:
                padded = np.pad(d, (0, max_len - len(d)), mode='edge')
                success_padded.append(padded)
            
            fail_padded = []
            for d in fail_dists:
                padded = np.pad(d, (0, max_len - len(d)), mode='edge')
                fail_padded.append(padded)
            
            if success_padded:
                success_mean = np.mean(success_padded, axis=0)
                success_std = np.std(success_padded, axis=0)
                steps = np.arange(len(success_mean))
                ax.plot(steps, success_mean, 'g-', linewidth=2.5, label='Successful (Mean)')
                ax.fill_between(steps, success_mean - success_std, success_mean + success_std,
                               alpha=0.2, color='g')
            
            if fail_padded:
                fail_mean = np.mean(fail_padded, axis=0)
                fail_std = np.std(fail_padded, axis=0)
                steps = np.arange(len(fail_mean))
                ax.plot(steps, fail_mean, 'r-', linewidth=2.5, label='Failed (Mean)')
                ax.fill_between(steps, fail_mean - fail_std, fail_mean + fail_std,
                               alpha=0.2, color='r')
            
            ax.axhline(y=0.3, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Distance to Goal (m)', fontsize=11)
            ax.set_title('Average Distance Trajectories', fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Success episode lengths
        ax = axes[0, 1]
        success_lengths = [len(ep.get('rewards', [])) for ep in successful_episodes]
        fail_lengths = [len(ep.get('rewards', [])) for ep in failed_episodes]
        
        if success_lengths or fail_lengths:
            data = [success_lengths, fail_lengths]
            labels = [f'Success\n(n={len(success_lengths)})', f'Failed\n(n={len(fail_lengths)})']
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax.set_ylabel('Episode Length (steps)', fontsize=11)
            ax.set_title('Episode Length Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Success rate by distance threshold
        ax = axes[1, 0]
        thresholds = np.linspace(0, 2.0, 21)
        success_rates = []
        
        if successful_episodes and 'final_distance' in successful_episodes[0]:
            all_finals = [ep.get('final_distance', np.inf) for ep in successful_episodes + failed_episodes]
            success_finals = [ep.get('final_distance', np.inf) for ep in successful_episodes]
            
            for thresh in thresholds:
                success_count = sum(1 for d in success_finals if d < thresh)
                total_count = sum(1 for d in all_finals if d < thresh)
                success_rates.append(success_count / max(total_count, 1) * 100)
        
        if success_rates:
            ax.plot(thresholds, success_rates, 'b-o', linewidth=2.5, markersize=6)
            ax.fill_between(thresholds, success_rates, alpha=0.2, color='b')
            ax.axvline(x=0.3, color='r', linestyle='--', linewidth=2, label='Standard threshold')
            ax.set_xlabel('Distance Threshold (m)', fontsize=11)
            ax.set_ylabel('Success Rate (%)', fontsize=11)
            ax.set_title('Success Rate vs Threshold', fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = "Episode Statistics:\n\n"
        stats_text += f"Successful Episodes: {len(successful_episodes)}\n"
        if successful_episodes and 'rewards' in successful_episodes[0]:
            success_rewards = [sum(ep['rewards']) for ep in successful_episodes]
            stats_text += f"  Avg Reward: {np.mean(success_rewards):.2f}\n"
            stats_text += f"  Max Reward: {np.max(success_rewards):.2f}\n\n"
        
        stats_text += f"Failed Episodes: {len(failed_episodes)}\n"
        if failed_episodes and 'rewards' in failed_episodes[0]:
            fail_rewards = [sum(ep['rewards']) for ep in failed_episodes]
            stats_text += f"  Avg Reward: {np.mean(fail_rewards):.2f}\n"
            stats_text += f"  Max Reward: {np.max(fail_rewards):.2f}\n"
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Success vs Failure Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self, episode_logs_dir: str, num_episodes: int = 5):
        """
        Generate all visualizations from episode logs.
        
        Args:
            episode_logs_dir: Directory containing episode log files
            num_episodes: Number of episodes to visualize
        """
        logs_dir = Path(episode_logs_dir)
        if not logs_dir.exists():
            print(f"Error: Episode logs directory not found: {logs_dir}")
            return
        
        print(f"\n{'='*60}")
        print("Generating ARM REACHING visualizations...")
        print(f"{'='*60}\n")
        
        # Load episodes
        episode_files = sorted(logs_dir.glob("episode_*.json"))[:num_episodes]
        
        if not episode_files:
            print(f"No episode logs found in {logs_dir}")
            return
        
        episodes = []
        for ep_file in episode_files:
            ep_data = self.load_episode_trajectory(str(ep_file))
            if ep_data:
                episodes.append(ep_data)
        
        print(f"Loaded {len(episodes)} episodes\n")
        
        # Generate individual episode visualizations
        for idx, ep_data in enumerate(episodes, 1):
            print(f"Generating visualizations for episode {idx}...")
            self.plot_2d_trajectory(ep_data, 
                                   title=f"Episode {idx}: 2D Trajectory",
                                   output_name=f"ep{idx:03d}_trajectory_2d")
            self.plot_3d_trajectory(ep_data,
                                   title=f"Episode {idx}: 3D Trajectory",
                                   output_name=f"ep{idx:03d}_trajectory_3d")
            self.plot_episode_metrics(ep_data,
                                     title=f"Episode {idx}: Detailed Metrics",
                                     output_name=f"ep{idx:03d}_metrics")
        
        # Generate comparison visualizations
        if len(episodes) > 1:
            successful = [ep for ep in episodes if ep.get('success', False)]
            failed = [ep for ep in episodes if not ep.get('success', False)]
            
            if successful and failed:
                self.plot_success_comparison(successful, failed)
        
        print(f"\n{'='*60}")
        print(f"✓ Visualizations saved to: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Visualize arm reaching episodes")
    parser.add_argument("--log_dir", type=str, default="logs/arm_episodes",
                       help="Directory containing episode logs")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to visualize")
    
    args = parser.parse_args()
    
    visualizer = ArmEpisodeVisualizer(args.output_dir)
    visualizer.generate_all_visualizations(args.log_dir, args.num_episodes)


if __name__ == "__main__":
    main()
