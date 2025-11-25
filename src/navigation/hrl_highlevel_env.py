# hrl_highlevel_env_improved.py

import gymnasium as gym
import numpy as np
from simple_navigation_env import SimpleNavigationEnv
from stable_baselines3 import PPO
import habitat_sim
import quaternion


class HRLHighLevelEnvImproved(gym.Env):
    """
    IMPROVED High-level environment with better subgoal generation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        low_level_model_path="models/lowlevel_curriculum_250k",
        subgoal_distance=5.0,
        option_horizon=50,
        debug=False,
    ):
        super().__init__()

        # Load low-level skill
        self.low_level = PPO.load(low_level_model_path)
        print("✓ Loaded low-level skill:", low_level_model_path)

        # Create low-level environment
        self.ll_env = SimpleNavigationEnv()
        self.sim: habitat_sim.Simulator = self.ll_env.sim
        self.pathfinder = self.sim.pathfinder

        # Parameters
        self.subgoal_radius = float(subgoal_distance)
        self.option_horizon = int(option_horizon)
        self.max_highlevel_steps = 20

        # Observation: [distance_to_main_goal, angle_to_main_goal]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi], dtype=np.float32),
            high=np.array([100.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: 8 directions on a circle
        self.action_space = gym.spaces.Discrete(8)

        self.current_step = 0
        self.main_goal = None
        self.prev_main_distance = None
        self.debug = debug
        
        # Track statistics
        self.failed_subgoal_count = 0
        self.successful_subgoal_count = 0

    def _quat_rotate(self, q_hab, v):
        """Rotate vector by quaternion"""
        q = np.quaternion(q_hab.w, q_hab.x, q_hab.y, q_hab.z)
        vq = np.quaternion(0.0, *v)
        rq = q * vq * q.inverse()
        return np.array([rq.x, rq.y, rq.z], dtype=np.float32)

    def _get_hl_obs(self):
        """Get high-level observation"""
        state = self.ll_env.agent.get_state()
        agent_pos = np.array(state.position, dtype=np.float32)

        # Distance to main goal
        to_goal = self.main_goal - agent_pos
        dist = float(np.linalg.norm(to_goal))

        if dist < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Agent forward direction
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward = self._quat_rotate(state.rotation, forward)
        forward[1] = 0.0

        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= np.linalg.norm(forward)

        # Direction to goal
        to_goal[1] = 0.0
        if np.linalg.norm(to_goal) < 1e-6:
            to_goal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            to_goal /= np.linalg.norm(to_goal)

        # Signed angle
        cross = forward[0] * to_goal[2] - forward[2] * to_goal[0]
        dot = forward[0] * to_goal[0] + forward[2] * to_goal[2]
        angle = float(np.arctan2(cross, dot))

        if np.isnan(angle):
            angle = 0.0
        if np.isnan(dist):
            dist = 50.0

        return np.array([dist, angle], dtype=np.float32)

    def _sample_subgoal(self, action):
        """
        IMPROVED subgoal sampling with multiple fallbacks
        """
        agent_pos = np.array(
            self.ll_env.agent.get_state().position, dtype=np.float32
        )
        
        # Get direction toward main goal
        to_main_goal = self.main_goal - agent_pos
        to_main_goal[1] = 0.0
        
        if np.linalg.norm(to_main_goal) > 1e-6:
            main_goal_dir = to_main_goal / np.linalg.norm(to_main_goal)
            main_goal_angle = np.arctan2(main_goal_dir[2], main_goal_dir[0])
        else:
            main_goal_angle = 0.0

        # 8 directions, but biased toward main goal
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        
        # Add bias: rotate actions to align roughly with goal direction
        action_angle = angles[int(action)] + main_goal_angle
        
        # Try multiple distances
        distances_to_try = [
            self.subgoal_radius,
            self.subgoal_radius * 0.7,  # Shorter
            self.subgoal_radius * 0.5,  # Even shorter
        ]
        
        for dist in distances_to_try:
            dx = dist * np.cos(action_angle)
            dz = dist * np.sin(action_angle)
            raw_target = agent_pos + np.array([dx, 0.0, dz], dtype=np.float32)
            
            # Snap to navmesh
            snapped = self.pathfinder.snap_point(raw_target)
            nav_target = np.array(snapped, dtype=np.float32)
            
            # Validate subgoal
            if not np.isnan(nav_target).any():
                movement_dist = np.linalg.norm(nav_target - agent_pos)
                if movement_dist > 1.5:  # At least 1.5m movement
                    # Check if pathfinder can find a path
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent_pos
                    path.requested_end = nav_target
                    
                    if self.pathfinder.find_path(path):
                        if path.geodesic_distance < 999.0:  # Valid path
                            if self.debug:
                                print(f"  ✓ Valid subgoal: {nav_target} (dist={movement_dist:.1f}m)")
                            self.successful_subgoal_count += 1
                            return nav_target
        
        # FALLBACK 1: Try random navigable points nearby
        for _ in range(10):
            random_pt = self.pathfinder.get_random_navigable_point()
            if np.linalg.norm(random_pt - agent_pos) > 2.0:
                if self.debug:
                    print(f"  → Fallback: random navigable point")
                self.failed_subgoal_count += 1
                return np.array(random_pt, dtype=np.float32)
        
        # FALLBACK 2: Move toward main goal with shorter distance
        direction = (self.main_goal - agent_pos)
        direction[1] = 0.0
        if np.linalg.norm(direction) > 1e-6:
            direction /= np.linalg.norm(direction)
            target = agent_pos + direction * 3.0  # Just 3m forward
            snapped = self.pathfinder.snap_point(target)
            if not np.isnan(snapped).any():
                if self.debug:
                    print(f"  → Fallback: toward main goal")
                self.failed_subgoal_count += 1
                return np.array(snapped, dtype=np.float32)
        
        # FALLBACK 3: Stay roughly where we are
        if self.debug:
            print(f"  ⚠️  All subgoal methods failed, minimal movement")
        self.failed_subgoal_count += 1
        return agent_pos + np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # Reset low-level env
        ll_obs, _ = self.ll_env.reset()
        agent_pos = np.array(
            self.ll_env.agent.get_state().position, dtype=np.float32
        )

        # Sample main goal (reasonably far)
        goal = None
        for _ in range(100):
            g = np.array(
                self.pathfinder.get_random_navigable_point(),
                dtype=np.float32,
            )
            dist = np.linalg.norm(g - agent_pos)
            if 8.0 <= dist <= 20.0:  # Reduced from 10m minimum
                goal = g
                break
        if goal is None:
            goal = g

        self.main_goal = goal
        self.current_step = 0
        self.prev_main_distance = float(np.linalg.norm(agent_pos - self.main_goal))
        self.failed_subgoal_count = 0
        self.successful_subgoal_count = 0

        if self.debug:
            print("\n[HL RESET]")
            print(f"Agent: {agent_pos}")
            print(f"Goal: {self.main_goal}")
            print(f"Distance: {self.prev_main_distance:.1f}m")

        return self._get_hl_obs(), {}

    def step(self, action):
        """Execute high-level action"""
        agent_pos_before = np.array(
            self.ll_env.agent.get_state().position, dtype=np.float32
        )
        
        # Get subgoal
        subgoal = self._sample_subgoal(action)
        
        if self.debug:
            print(f"\n=== HL Step {self.current_step + 1} ===")
            print(f"Action: {action}")
            print(f"Subgoal: {subgoal}")

        # Execute low-level skill
        self.ll_env.goal_position = subgoal
        self.ll_env.prev_distance = float(
            np.linalg.norm(agent_pos_before - subgoal)
        )

        ll_steps_taken = 0
        for i in range(self.option_horizon):
            obs_ll = self.ll_env._get_obs()
            obs_ll = np.nan_to_num(obs_ll, nan=0.0)
            ll_action, _ = self.low_level.predict(obs_ll, deterministic=True)
            _, _, done_ll, trunc_ll, _ = self.ll_env.step(ll_action)
            ll_steps_taken += 1
            
            if done_ll or trunc_ll:
                break

        agent_pos_after = np.array(
            self.ll_env.agent.get_state().position, dtype=np.float32
        )
        
        movement = np.linalg.norm(agent_pos_after - agent_pos_before)
        
        # Calculate reward
        new_dist = float(np.linalg.norm(agent_pos_after - self.main_goal))
        if np.isnan(new_dist):
            new_dist = 50.0

        progress = self.prev_main_distance - new_dist
        
        # IMPROVED REWARD SHAPING
        reward = 10.0 * progress  # Strong progress reward
        
        # Bonus for significant progress
        if progress > 2.0:
            reward += 5.0
        elif progress > 1.0:
            reward += 2.0
        
        # Penalty for no movement
        if movement < 0.5:
            reward -= 1.0
        
        # Small time penalty
        reward -= 0.05

        done = False
        if new_dist < 0.6:
            reward += 50.0  # Huge success bonus!
            done = True

        truncated = self.current_step >= self.max_highlevel_steps

        if self.debug:
            print(f"Movement: {movement:.2f}m")
            print(f"Distance: {self.prev_main_distance:.1f}m → {new_dist:.1f}m")
            print(f"Progress: {progress:.2f}m")
            print(f"Reward: {reward:.2f}")

        self.prev_main_distance = new_dist
        self.current_step += 1

        return self._get_hl_obs(), reward, done, truncated, {
            "main_distance": new_dist,
            "movement": movement,
            "progress": progress,
        }

    def close(self):
        if self.debug:
            total = self.successful_subgoal_count + self.failed_subgoal_count
            if total > 0:
                success_rate = (self.successful_subgoal_count / total) * 100
                print(f"\nSubgoal generation stats: {success_rate:.0f}% valid ({self.successful_subgoal_count}/{total})")
        self.ll_env.close()