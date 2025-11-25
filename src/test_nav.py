"""
Visualize Fetch robot navigation in Skokloster Castle using trained HRL model.

This script demonstrates how to:
1. Load a trained low-level navigation agent (PPO)
2. Set up Habitat-Sim with the Fetch robot and visual sensors
3. Run the agent and visualize the results with RGB camera output
4. Save video frames of the navigation episode

Usage:
    python test_nav.py [--no-display] [--save-video] [--episodes N]
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from stable_baselines3 import PPO
import habitat_sim
import quaternion  # numpy-quaternion


class FetchNavigationVisualizer:
    """
    Habitat-Sim environment with Fetch robot and visual rendering.
    Integrates with trained low-level navigation policy.
    """
    
    def __init__(self, 
                 model_path=None,
                 scene_path="habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
                 enable_video=False):
        """
        Args:
            model_path: Path to trained PPO model (.zip file)
            scene_path: Path to scene GLB file
            enable_video: Whether to save video frames
        """
        self.scene_path = scene_path
        self.enable_video = enable_video
        self.video_frames = []
        
        # Load trained model
        if model_path is None:
            model_path = "" # path to lowlevel_curriculum_250k.zip
            #model_path = self._find_model()
        
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        
        # Initialize simulator
        self._setup_simulator()
        
        print("✓ FetchNavigationVisualizer initialized")
    
    def _find_model(self):
        """Auto-detect trained model file."""

        search_paths = [
            "navigation/models/lowlevel_curriculum_250k.zip",
            "src/navigation/models/lowlevel_curriculum_250k.zip",
            "models/lowlevel_curriculum_250k.zip",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to latest checkpoint
        checkpoint_dirs = [
            "navigation/models/checkpoints",
            "src/navigation/models/checkpoints",
            "models/checkpoints"
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                import glob
                checkpoints = glob.glob(f"{checkpoint_dir}/*.zip")
                if checkpoints:
                    latest = max(checkpoints, key=os.path.getctime)
                    return latest
        
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    def _setup_simulator(self):
        """Initialize Habitat-Sim with Fetch robot and RGB sensor."""
        
        # Backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = True  # Enable physics for robot
        backend_cfg.gpu_device_id = 0  # Use GPU if available, -1 for CPU
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Add RGB sensor for visualization
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]  # Height x Width
        rgb_sensor_spec.position = [0.0, 1.25, 0.0]  # Relative to agent
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
        
        # Robot body parameters (Fetch robot approximate dimensions)
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.2
        
        # Movement actions for navigation
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", 
                habitat_sim.agent.ActuationSpec(amount=0.5)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", 
                habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", 
                habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        self.pathfinder = self.sim.pathfinder
        
        # Navigation parameters
        self.max_steps = 300
        self.goal_threshold = 0.5
        self.current_step = 0
        self.goal_position = None
        self.prev_distance = None
    
    def _quat_rotate(self, q_hab, v):
        """Rotate vector by quaternion."""
        q = np.quaternion(q_hab.w, q_hab.x, q_hab.y, q_hab.z)
        vq = np.quaternion(0.0, v[0], v[1], v[2])
        rq = q * vq * q.inverse()
        return np.array([rq.x, rq.y, rq.z], dtype=np.float32)
    
    def _get_observation(self):
        """
        Get observation for the trained model.
        Returns: [distance_to_goal, relative_angle_to_goal]
        """
        state = self.agent.get_state()
        agent_pos = state.position
        quat = state.rotation
        
        # Distance to goal
        distance = np.linalg.norm(agent_pos - self.goal_position)
        
        if distance < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Agent forward direction (in world frame)
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward = self._quat_rotate(quat, forward)
        forward[1] = 0.0
        
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= np.linalg.norm(forward)
        
        # Direction to goal
        to_goal = self.goal_position - agent_pos
        to_goal[1] = 0.0
        
        if np.linalg.norm(to_goal) < 1e-6:
            to_goal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            to_goal /= np.linalg.norm(to_goal)
        
        # Compute signed angle
        cross = forward[0] * to_goal[2] - forward[2] * to_goal[0]
        dot = forward[0] * to_goal[0] + forward[2] * to_goal[2]
        angle = np.arctan2(cross, dot)
        
        # Safety checks
        if np.isnan(angle):
            angle = 0.0
        if np.isnan(distance):
            distance = 50.0
        
        return np.array([distance, angle], dtype=np.float32)
    
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
    
    def reset(self, min_goal_dist=2.0, max_goal_dist=8.0):
        """Reset episode with new start and goal positions."""
        
        # Sample agent start position
        agent_pos = self.pathfinder.get_random_navigable_point()
        
        # Sample goal within distance range
        for _ in range(50):
            goal = self.pathfinder.get_random_navigable_point()
            dist = np.linalg.norm(agent_pos - goal)
            if min_goal_dist <= dist <= max_goal_dist:
                self.goal_position = goal
                break
        else:
            self.goal_position = goal
        
        # Set agent state
        state = habitat_sim.AgentState()
        state.position = agent_pos
        state.rotation = habitat_sim.utils.quat_from_angle_axis(
            0.0, np.array([0.0, 1.0, 0.0])
        )
        self.agent.set_state(state)
        
        self.current_step = 0
        self.prev_distance = np.linalg.norm(agent_pos - self.goal_position)
        self.video_frames = []
        
        obs = self._get_observation()
        print(f"\n▶ Episode Start:")
        print(f"  Start position: {agent_pos}")
        print(f"  Goal position:  {self.goal_position}")
        print(f"  Initial distance: {obs[0]:.2f}m")
        
        return obs
    
    def step(self, action):
        """
        Execute action in environment.
        
        Args:
            action: 0=NO-OP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        
        Returns:
            obs, distance, done, info
        """
        # Execute action
        if action == 1:
            self.agent.act("move_forward")
        elif action == 2:
            self.agent.act("turn_left")
        elif action == 3:
            self.agent.act("turn_right")
        # action 0 = NO-OP
        
        self.current_step += 1
        
        # Get new observation
        obs = self._get_observation()
        distance = float(obs[0])
        
        # Check termination
        done = distance < self.goal_threshold
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance": distance,
            "step": self.current_step,
            "success": done,
            "truncated": truncated
        }
        
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
        
        action_names = ["NO-OP", "FORWARD", "LEFT", "RIGHT"]
        
        while not done:
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            
            # Step environment
            obs, distance, done, info = self.step(action)
            
            # Get and process RGB frame
            rgb_frame = self._get_rgb_observation()
            
            if rgb_frame is not None:
                # Add info overlay
                frame = self._add_overlay(
                    rgb_frame, 
                    action_names[action],
                    info['step'],
                    distance
                )
                
                if self.enable_video:
                    self.video_frames.append(frame)
                
                if display:
                    cv2.imshow("Fetch Navigation - Skokloster Castle", frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
            
            # Print progress
            if info['step'] % 10 == 0:
                print(f"  Step {info['step']:3d}: {action_names[action]:7s} | Distance: {distance:.2f}m")
        
        # Episode complete
        success = info['success']
        final_dist = info['distance']
        num_steps = info['step']
        
        print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"  Steps: {num_steps}")
        print(f"  Final distance: {final_dist:.2f}m")
        
        if display:
            cv2.destroyAllWindows()
        
        return success, num_steps, final_dist
    
    def _add_overlay(self, frame, action, step, distance):
        """Add information overlay to frame."""
        frame = frame.copy()
        
        # Add semi-transparent overlay bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Step: {step}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {action}", (10, 50), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 75), font, 0.6, (255, 255, 255), 2)
        
        # Add goal indicator
        color = (0, 255, 0) if distance < self.goal_threshold else (0, 165, 255)
        cv2.circle(frame, (frame.shape[1] - 40, 40), 20, color, -1)
        cv2.putText(frame, "GOAL", (frame.shape[1] - 70, 75), font, 0.4, (255, 255, 255), 1)
        
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
        print(f"\n{'='*60}")
        print(f"Fetch Robot Navigation Evaluation")
        print(f"Scene: Skokloster Castle")
        print(f"Episodes: {num_episodes}")
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
                video_name = f"videos/fetch_nav_ep{ep+1:02d}.mp4"
                self.save_video(video_name)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        successes = sum(1 for r in results if r['success'])
        success_rate = successes / num_episodes * 100
        
        successful_eps = [r for r in results if r['success']]
        avg_steps = np.mean([r['steps'] for r in successful_eps]) if successful_eps else 0
        
        print(f"Success Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
        if successful_eps:
            print(f"Average Steps (successful): {avg_steps:.1f}")
        print(f"{'='*60}\n")
        
        return results
    
    def close(self):
        """Clean up simulator."""
        self.sim.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Fetch robot navigation in Skokloster Castle"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to trained model (auto-detected if not specified)"
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
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = FetchNavigationVisualizer(
            model_path=args.model,
            scene_path=args.scene,
            enable_video=args.save_video
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