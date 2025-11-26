"""
Visualize Fetch robot navigation in Skokloster Castle using trained HRL model.

This script demonstrates how to:
1. Load trained high-level and low-level navigation agents (HRL with PPO)
2. Set up Habitat-Sim with the Fetch robot and visual sensors
3. Run the HRL agent (high-level planner + low-level controller)
4. Visualize results with RGB camera output and save video frames

Usage:
    python test_nav.py [--no-display] [--save-video] [--episodes N]
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from stable_baselines3 import PPO
import habitat_sim
import quaternion  # numpy-quaternion

# Add navigation directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'navigation'))
from navigation.hrl_highlevel_env import HRLHighLevelEnvImproved


class FetchNavigationVisualizer:
    """
    Habitat-Sim environment with Fetch robot and visual rendering.
    Integrates with trained HRL (Hierarchical Reinforcement Learning) system:
    - High-level: Subgoal planner (navigates through waypoints)
    - Low-level: Motor controller (executes navigation to subgoals)
    """
    
    def __init__(self, 
                 low_level_model_path=None,
                 high_level_model_path=None,
                 scene_path="habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
                 enable_video=False,
                 debug=False,
                 use_low_level_only=False):
        """
        Args:
            low_level_model_path: Path to trained low-level PPO model (.zip file)
            high_level_model_path: Path to trained high-level PPO model (.zip file)
            scene_path: Path to scene GLB file
            enable_video: Whether to save video frames
            debug: Print detailed HRL execution info
            use_low_level_only: If True, use only low-level agent directly
        """
        self.scene_path = scene_path
        self.enable_video = enable_video
        self.video_frames = []
        self.debug = debug
        self.use_low_level_only = use_low_level_only
        
        # Find models if not specified
        if low_level_model_path is None:
            low_level_model_path = self._find_model("lowlevel")
        if high_level_model_path is None:
            high_level_model_path = self._find_model("highlevel")
        
        print(f"Loading low-level model from: {low_level_model_path}")
        print(f"Loading high-level model from: {high_level_model_path}")
        
        # Create HRL environment (contains both levels)
        self.hrl_env = HRLHighLevelEnvImproved(
            low_level_model_path=low_level_model_path,
            subgoal_distance=5.0,
            option_horizon=50,
            debug=debug
        )
        
        # Load high-level model
        self.high_level_model = PPO.load(high_level_model_path)
        
        # Get references to simulator components
        self.sim = self.hrl_env.sim
        self.agent = self.hrl_env.ll_env.agent
        self.pathfinder = self.hrl_env.pathfinder
        
        # Add RGB sensor for visualization
        self._add_rgb_sensor()
        
        # Navigation parameters
        self.max_steps = 500 if not use_low_level_only else 500
        self.goal_threshold = 0.6
        self.current_step = 0
        self.main_goal = None
        
        mode = "Low-Level Only" if use_low_level_only else "HRL"
        print(f"✓ FetchNavigationVisualizer initialized ({mode} mode)")
    
    def _find_model(self, model_type):
        """Auto-detect trained model file."""
        if model_type == "lowlevel":
            search_paths = [
                "navigation/models/lowlevel_curriculum_250k.zip",
                "src/navigation/models/lowlevel_curriculum_250k.zip",
                "models/lowlevel_curriculum_250k.zip",
            ]
            checkpoint_dirs = [
                "navigation/models/lowlevel_checkpoints",
                "src/navigation/models/lowlevel_checkpoints",
            ]
        else:  # highlevel
            search_paths = [
                "navigation/models/highlevel_manager_final.zip",
                "src/navigation/models/highlevel_manager_final.zip",
                "models/highlevel_manager_final.zip",
            ]
            checkpoint_dirs = [
                "navigation/models/hl_improved",
                "src/navigation/models/hl_improved",
                "navigation/models/hl_checkpoints",
                "src/navigation/models/hl_checkpoints",
            ]
        
        # Try direct paths first
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to latest checkpoint
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                import glob
                checkpoints = glob.glob(f"{checkpoint_dir}/*.zip")
                if checkpoints:
                    latest = max(checkpoints, key=os.path.getctime)
                    return latest
        
        raise FileNotFoundError(f"No trained {model_type} model found. Please train models first.")
    
    def _add_rgb_sensor(self):
        """Add RGB camera sensor to existing agent for visualization."""
        # Save current agent state
        agent_state = self.agent.get_state()
        
        # Close the existing simulator
        self.sim.close()
        
        # Create backend configuration matching the existing simulator
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = False
        backend_cfg.gpu_device_id = -1  # CPU rendering
        
        # Create new agent config with RGB sensor
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # RGB sensor specification
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]  # Height x Width
        rgb_sensor_spec.position = [0.0, 1.25, 0.0]  # Relative to agent (Fetch robot height)
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
        
        # Copy action space and body parameters from existing agent
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.5)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.1
        
        # Create new simulator with RGB sensor
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        
        # Update references
        self.agent = self.sim.get_agent(0)
        self.hrl_env.sim = self.sim
        self.hrl_env.ll_env.sim = self.sim
        self.hrl_env.ll_env.agent = self.agent
        self.hrl_env.pathfinder = self.sim.pathfinder
        self.pathfinder = self.sim.pathfinder
        
        # Restore agent state
        self.agent.set_state(agent_state)
    
    def _get_observation(self):
        """
        Get observation (high-level or low-level depending on mode).
        Returns: [distance_to_goal, relative_angle_to_goal]
        """
        if self.use_low_level_only:
            return self.hrl_env.ll_env._get_obs()
        else:
            return self.hrl_env._get_hl_obs()
    
    def _get_rgb_observation(self):
        """Get RGB camera observation for visualization."""
        observations = self.sim.get_sensor_observations()
        rgb = observations.get("color_sensor")
        
        if rgb is not None:
            # Convert from RGB to BGR for OpenCV
            rgb = rgb[:, :, :3]  # Remove alpha channel if present
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        return None
    
    def reset(self):
        """Reset episode with new start and goal positions."""
        if self.use_low_level_only:
            obs, info = self.hrl_env.ll_env.reset()
            self.main_goal = self.hrl_env.ll_env.goal_position
        else:
            obs, info = self.hrl_env.reset()
            self.main_goal = self.hrl_env.main_goal
        
        self.current_step = 0
        self.video_frames = []
        
        agent_pos = self.agent.get_state().position
        
        mode = "Low-Level" if self.use_low_level_only else "HRL"
        print(f"\n▶ Episode Start ({mode}):")
        print(f"  Start position: {agent_pos}")
        print(f"  Goal: {self.main_goal}")
        print(f"  Initial distance: {obs[0]:.2f}m")
        if self.use_low_level_only:
            print(f"  Action space: 4 actions (NO-OP, FORWARD, LEFT, RIGHT)")
        else:
            print(f"  High-level action space: 8 directions")
        
        return obs
    
    def step(self, action):
        """
        Execute action (high-level or low-level depending on mode).
        
        Args:
            action: High-level (0-7) or low-level (0-3) action
        
        Returns:
            obs, distance, done, info
        """
        if self.use_low_level_only:
            obs, reward, done, truncated, info = self.hrl_env.ll_env.step(action)
        else:
            obs, reward, done, truncated, info = self.hrl_env.step(action)
        
        self.current_step += 1
        distance = info.get("main_distance" if not self.use_low_level_only else "distance", obs[0])
        
        info.update({
            "step": self.current_step,
            "success": done and not truncated,
            "truncated": truncated,
            "reward": reward
        })
        
        return obs, distance, done or truncated, info
    
    def visualize_episode(self, display=True):
        """
        Run one episode with visualization.
        
        Args:
            display: Whether to display frames in real-time
        
        Returns:
            success, num_steps, final_distance
        """
        obs = self.reset()
        done = False
        
        if self.use_low_level_only:
            action_names = ["NO-OP", "FWD", "LEFT", "RIGHT"]
        else:
            action_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        
        total_reward = 0.0
        
        while not done:
            # Get action from appropriate model
            if self.use_low_level_only:
                action, _ = self.hrl_env.low_level.predict(obs, deterministic=True)
            else:
                action, _ = self.high_level_model.predict(obs, deterministic=True)
            
            action = int(action)
            
            agent_pos_before = self.agent.get_state().position.copy()
            
            # Step environment
            obs, distance, done, info = self.step(action)
            
            agent_pos_after = self.agent.get_state().position
            total_reward += info['reward']
            
            # Get and process RGB frame
            rgb_frame = self._get_rgb_observation()
            
            if rgb_frame is not None:
                # Add info overlay
                frame = self._add_overlay(
                    rgb_frame, 
                    action_names[action],
                    info['step'],
                    distance,
                    info
                )
                
                if self.enable_video:
                    self.video_frames.append(frame)
                
                if display:
                    cv2.imshow("Fetch Navigation - Skokloster Castle", frame)
                    wait_time = 50 if not self.use_low_level_only else 10
                    key = cv2.waitKey(wait_time)
                    if key == ord('q'):
                        break
            
            # Print progress
            movement = np.linalg.norm(agent_pos_after - agent_pos_before)
            progress = info.get('progress', 0.0)
            prefix = "LL" if self.use_low_level_only else "HL"
            print(f"  {prefix} Step {info['step']:3d}: {action_names[action]:5s} | "
                  f"Dist: {distance:.2f}m | Movement: {movement:.2f}m | "
                  f"Progress: {progress:+.2f}m | Reward: {info['reward']:+.2f}")
        
        # Episode complete
        success = info['success']
        final_dist = distance
        num_steps = info['step']
        
        print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"  Steps: {num_steps}")
        print(f"  Final distance: {final_dist:.2f}m")
        print(f"  Total reward: {total_reward:.2f}")
        
        if display:
            cv2.destroyAllWindows()
        
        return success, num_steps, final_dist
    
    def _add_overlay(self, frame, action, step, distance, info=None):
        """Add information overlay to frame."""
        frame = frame.copy()
        
        # Add semi-transparent overlay bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        prefix = "LL" if self.use_low_level_only else "HL"
        cv2.putText(frame, f"{prefix} Step: {step}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {action}", (10, 50), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 75), font, 0.6, (255, 255, 255), 2)
        
        # Add movement and reward if available
        if info:
            movement = info.get('movement', 0.0)
            reward = info.get('reward', 0.0)
            cv2.putText(frame, f"Move: {movement:.1f}m | R: {reward:+.1f}", 
                       (10, 100), font, 0.5, (200, 200, 200), 1)
        
        # Add goal indicator
        color = (0, 255, 0) if distance < self.goal_threshold else (0, 165, 255)
        cv2.circle(frame, (frame.shape[1] - 40, 40), 20, color, -1)
        cv2.putText(frame, "GOAL", (frame.shape[1] - 70, 75), font, 0.4, (255, 255, 255), 1)
        
        # Add mode indicator
        mode_text = "LL" if self.use_low_level_only else "HRL"
        cv2.putText(frame, mode_text, (frame.shape[1] - 60, 100), font, 0.4, (100, 255, 100), 1)
        
        return frame
    
    def save_video(self, filename="fetch_navigation.mp4", fps=10):
        """Save recorded frames as video."""
        if not self.video_frames:
            print("No frames to save")
            return
        
        # Ensure output directory exists
        output_dir = Path(filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        height, width = self.video_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in self.video_frames:
            out.write(frame)
        
        out.release()
        print(f"✓ Video saved to: {filename}")
    
    def run_evaluation(self, num_episodes=5, display=True, save_videos=False):
        """Run multiple episodes and report statistics."""
        mode = "Low-Level" if self.use_low_level_only else "HRL"
        print(f"\n{'='*60}")
        print(f"Fetch Robot Navigation Evaluation ({mode})")
        print(f"Scene: Skokloster Castle")
        print(f"Episodes: {num_episodes}")
        if not self.use_low_level_only:
            print(f"High-level: Subgoal planner (8 directions)")
            print(f"Low-level: Motor controller (discrete actions)")
        else:
            print(f"Mode: Low-level motor controller only (4 actions)")
        print(f"{'='*60}")
        
        results = []
        
        for ep in range(num_episodes):
            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
            
            success, steps, final_dist = self.visualize_episode(display=display)
            results.append({
                'success': success,
                'steps': steps,
                'final_distance': final_dist
            })
            
            if save_videos and self.enable_video:
                prefix = "ll" if self.use_low_level_only else "hrl"
                video_name = f"videos/fetch_{prefix}_nav_ep{ep+1:02d}.mp4"
                self.save_video(video_name)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"{mode} EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        successes = sum(1 for r in results if r['success'])
        success_rate = successes / num_episodes * 100
        
        successful_eps = [r for r in results if r['success']]
        avg_steps = np.mean([r['steps'] for r in successful_eps]) if successful_eps else 0
        
        print(f"Success Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
        if successful_eps:
            step_type = "steps" if self.use_low_level_only else "high-level steps"
            print(f"Average {step_type} (successful): {avg_steps:.1f}")
            if not self.use_low_level_only:
                print(f"  (Each HL step = up to 50 low-level actions)")
        print(f"{'='*60}\n")
        
        return results
    
    def close(self):
        """Clean up simulator and HRL environment."""
        self.hrl_env.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Fetch robot navigation in Skokloster Castle"
    )
    parser.add_argument(
        "--low-level-model", 
        type=str, 
        default=None,
        help="Path to trained low-level model (auto-detected if not specified)"
    )
    parser.add_argument(
        "--high-level-model", 
        type=str, 
        default=None,
        help="Path to trained high-level model (auto-detected if not specified)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        help="Path to scene GLB file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display visualization window"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos of episodes"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed execution information"
    )
    parser.add_argument(
        "--low-level-only",
        action="store_true",
        help="Use only low-level agent (no hierarchical planning)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = FetchNavigationVisualizer(
            low_level_model_path=args.low_level_model,
            high_level_model_path=args.high_level_model,
            scene_path=args.scene,
            enable_video=args.save_video,
            debug=args.debug,
            use_low_level_only=args.low_level_only
        )
        
        # Run evaluation
        visualizer.run_evaluation(
            num_episodes=args.episodes,
            display=not args.no_display,
            save_videos=args.save_video
        )
        
        # Cleanup
        visualizer.close()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()