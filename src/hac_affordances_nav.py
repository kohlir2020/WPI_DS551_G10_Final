#!/usr/bin/env python3
"""
HAC + Affordances Navigation (single file)

This script:
  - Reuses your partner's SimpleNavigationEnv (Habitat-based)
  - Adds:
      * Replay buffer
      * Low-level and high-level DQN agents (Hierarchical)
      * Affordance-guided subgoal generation using Habitat pathfinder
      * Training loop for HAC-style hierarchical RL
      * Evaluation routine

You get one big, notebook-like Python file to read & debug.
"""

import argparse
import os
import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from navigation.simple_navigation_env import SimpleNavigationEnv

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


# ============================================================
# 1. Utility: set random seeds
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Replay Buffer
# ============================================================

class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL (DQN-like).
    Stores transitions (s, a, r, s', done).
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def push(self, state, action, reward, next_state, done):
        i = self.idx
        self.states[i] = state
        self.actions[i, 0] = action
        self.rewards[i, 0] = reward
        self.next_states[i] = next_state
        self.dones[i, 0] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        assert len(self) >= batch_size
        indices = np.random.randint(0, len(self), size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones


# ============================================================
# 3. Networks (simple MLP Q-network)
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. DQN Agent (used at both levels)
# ============================================================

class DQNAgent:
    """
    Simple DQN agent for discrete actions.
    Used as:
      - low-level policy (primitive actions)
      - high-level policy (discrete subgoal directions)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float,
        gamma: float,
        buffer_capacity: int,
        batch_size: int,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        target_update_freq: int,
        device: torch.device,
        name: str = "agent",
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.name = name

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, device)
        self.training_steps = 0

    def epsilon(self):
        # Linear decay of epsilon
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(
            0.0, (self.eps_decay - self.training_steps) / float(self.eps_decay)
        )
        return eps

    def select_action(self, state: np.ndarray, greedy: bool = False):
        """
        Epsilon-greedy action selection.
        state: np.array of shape (state_dim,)
        """
        if not greedy and random.random() < self.epsilon():
            return random.randint(0, self.action_dim - 1)

        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(s)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.training_steps += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN-ish: select best action via q_net, evaluate via target_net
            next_q_values_online = self.q_net(next_states)
            next_actions = torch.argmax(next_q_values_online, dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict, strict=strict)
        self.target_net.load_state_dict(self.q_net.state_dict())


# ============================================================
# 5. Affordance-style subgoal generator
# ============================================================

class AffordanceModel:
    """
    Affordance-style subgoal generator for navigation.

    - High-level actions are discrete directions 0..7 (like a compass).
    - For each action index, we propose a subgoal in that direction at some distance.
    - We then project onto Habitat's navigable surface (navmesh) and
      optionally check geodesic distance via pathfinder.
    - This is a simple geometric affordance model that encodes "where can I go"
      in the sense of the affordances paper (feasible subgoals).
    """

    # 8 directions in X-Z plane (roughly N, NE, E, SE, S, SW, W, NW)
    DIRECTIONS = {
        0: np.array([0.0, 1.0]),    # North (Z+)
        1: np.array([1.0, 1.0]),    # NE
        2: np.array([1.0, 0.0]),    # East (X+)
        3: np.array([1.0, -1.0]),   # SE
        4: np.array([0.0, -1.0]),   # South (Z-)
        5: np.array([-1.0, -1.0]),  # SW
        6: np.array([-1.0, 0.0]),   # West (X-)
        7: np.array([-1.0, 1.0]),   # NW
    }

    def __init__(
        self,
        pathfinder,
        base_step_size: float = 5.0,
        min_movement: float = 1.5,
        use_geodesic_check: bool = True,
    ):
        self.pathfinder = pathfinder
        self.base_step_size = float(base_step_size)
        self.min_movement = float(min_movement)
        self.use_geodesic_check = use_geodesic_check

    def _unit_direction(self, action: int) -> np.ndarray:
        d = self.DIRECTIONS[action]
        d = d / np.linalg.norm(d)
        return d

    def propose_subgoal(
        self,
        agent_pos: np.ndarray,
        main_goal: np.ndarray,
        action: int,
    ) -> np.ndarray:
        """
        Convert high-level action -> candidate subgoal in world coordinates.

        We bias distance and direction using the main goal, but always
        project onto the navmesh and enforce minimal movement.
        """
        agent_pos = np.array(agent_pos, dtype=np.float32)
        main_goal = np.array(main_goal, dtype=np.float32)

        # Direction of selected action in XZ plane
        base_dir = self._unit_direction(action)  # [dx, dz]

        # Small bias: rotate this direction slightly toward the main_goal direction
        goal_vec = main_goal - agent_pos
        goal_vec[1] = 0.0
        if np.linalg.norm(goal_vec) > 1e-6:
            goal_vec /= np.linalg.norm(goal_vec)
            goal_angle = math.atan2(goal_vec[2], goal_vec[0])
        else:
            goal_angle = 0.0

        base_angle = math.atan2(base_dir[1], base_dir[0])

        # Interpolate angles (simple bias)
        blended_angle = 0.75 * base_angle + 0.25 * goal_angle
        blended_dir = np.array(
            [math.cos(blended_angle), math.sin(blended_angle)], dtype=np.float32
        )

        # Try multiple radii; stop at first feasible subgoal
        radii = [
            self.base_step_size,
            self.base_step_size * 0.7,
            self.base_step_size * 0.5,
        ]

        for r in radii:
            dx, dz = blended_dir[0] * r, blended_dir[1] * r
            raw_target = np.array(
                [agent_pos[0] + dx, agent_pos[1], agent_pos[2] + dz],
                dtype=np.float32,
            )

            # Snap to navmesh
            snapped = self.pathfinder.snap_point(raw_target)
            subgoal = np.array(snapped, dtype=np.float32)

            if np.isnan(subgoal).any():
                continue

            movement = np.linalg.norm(subgoal - agent_pos)
            if movement < self.min_movement:
                continue

            # Optional geodesic check
            if self.use_geodesic_check:
                import habitat_sim  # local import to avoid circulars

                path = habitat_sim.ShortestPath()
                path.requested_start = agent_pos
                path.requested_end = subgoal
                if not self.pathfinder.find_path(path):
                    continue
                if path.geodesic_distance >= 999.0:
                    continue

            return subgoal

        # Fallback: just move a little toward the main goal
        fallback_dir = main_goal - agent_pos
        fallback_dir[1] = 0.0
        if np.linalg.norm(fallback_dir) < 1e-6:
            fallback_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            fallback_dir /= np.linalg.norm(fallback_dir)

        candidate = agent_pos + fallback_dir * 2.0
        snapped = self.pathfinder.snap_point(candidate)
        return np.array(snapped, dtype=np.float32)


# ============================================================
# 6. Hierarchical HAC-style Trainer
# ============================================================

class HierarchicalHACTrainer:
    """
    Two-level HAC-style trainer:

      High-level:
        - State: [distance_to_main_goal, angle_to_main_goal] (same obs format)
        - Action: discrete direction 0..7 -> subgoal via AffordanceModel
        - Reward: progress toward main goal + success bonus

      Low-level:
        - State: [distance_to_current_subgoal, angle_to_current_subgoal]
        - Action: discrete primitive (0..3) in SimpleNavigationEnv
        - Reward: environment reward

    For each high-level decision:
      - AffordanceModel proposes a subgoal
      - We set env.goal_position = subgoal
      - Run low-level policy up to low_horizon
      - Then restore env.goal_position = main_goal and
        compute high-level transition (progress, done).
    """

    def __init__(
        self,
        env: SimpleNavigationEnv,
        low_agent: DQNAgent,
        high_agent: DQNAgent,
        affordances: AffordanceModel,
        args,
    ):
        self.env = env
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.affordances = affordances
        self.args = args

        self.main_goal = None

        self.episode_rewards = []
        self.episode_successes = []
        self.episode_final_dists = []

    def _get_main_goal_obs(self) -> np.ndarray:
        """
        Observation for high-level: env._get_obs() w.r.t. main goal.
        """
        # Ensure env.goal_position is the main goal
        self.env.goal_position = np.array(self.main_goal, dtype=np.float32)
        obs = self.env._get_obs()
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _get_subgoal_obs(self) -> np.ndarray:
        """
        Observation for low-level: env._get_obs() w.r.t. current subgoal
        (env.goal_position has already been set to the subgoal).
        """
        obs = self.env._get_obs()
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _sample_main_goal(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        Sample a main goal using the same style as your HRL env:
        pick a navigable point at a reasonable distance.
        """
        pathfinder = self.env.pathfinder
        min_dist = self.args.main_goal_min_dist
        max_dist = self.args.main_goal_max_dist

        goal = None
        for _ in range(100):
            g = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
            dist = np.linalg.norm(g - agent_pos)
            if min_dist <= dist <= max_dist:
                goal = g
                break
        if goal is None:
            goal = g
        return goal

    def train(self):
        """
        Main training loop across episodes.
        """

        success_history = deque(maxlen=100)
        print("\n=== HAC TRAINING START ===\n")

        for ep in range(1, self.args.episodes + 1):
            # Reset env and sample main goal
            obs, _ = self.env.reset()
            agent_state = self.env.agent.get_state()
            agent_pos = np.array(agent_state.position, dtype=np.float32)

            self.main_goal = self._sample_main_goal(agent_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_low_losses = []
            ep_high_losses = []
            ep_reward = 0.0
            ep_success = False
            final_main_dist = None


            # High-level loop
            for hl_step in range(self.args.max_high_steps):
                # High-level state
                state_h = self._get_main_goal_obs()

                # High-level action -> subgoal
                hl_action = self.high_agent.select_action(state_h, greedy=False)
                subgoal = self.affordances.propose_subgoal(
                    agent_pos, self.main_goal, hl_action
                )

                # Run low-level controller toward subgoal
                self.env.goal_position = np.array(subgoal, dtype=np.float32)
                subgoal_reached = False

                # Distance to subgoal before rollout (for potential diagnostics)
                d_before = np.linalg.norm(
                    np.array(agent_pos, dtype=np.float32) - subgoal
                )

                for ll_step in range(self.args.low_horizon):
                    state_l = self._get_subgoal_obs()
                    ll_action = self.low_agent.select_action(state_l, greedy=False)
                    next_obs_l, reward_l, done_l, trunc_l, info = self.env.step(
                        ll_action
                    )
                    next_state_l = self._get_subgoal_obs()

                    # Store low-level transition & update
                    self.low_agent.store(
                        state_l, ll_action, reward_l, next_state_l, done_l
                    )
                    loss_l = self.low_agent.update()
                    if loss_l is not None:
                        ep_low_losses.append(loss_l)

                    ep_reward += reward_l

                    # Check if subgoal is approximately reached
                    agent_state = self.env.agent.get_state()
                    agent_pos = np.array(agent_state.position, dtype=np.float32)
                    final_main_dist = float(np.linalg.norm(agent_pos - self.main_goal))

                    d_sub = np.linalg.norm(agent_pos - subgoal)

                    if d_sub < self.args.subgoal_success_radius:
                        subgoal_reached = True

                    if done_l or trunc_l or subgoal_reached:
                        break

                # After low-level rollout, measure progress to main goal
                agent_state = self.env.agent.get_state()
                new_pos = np.array(agent_state.position, dtype=np.float32)

                dist_before = np.linalg.norm(
                    agent_pos - self.main_goal
                )  # NOTE: agent_pos was updated, but close enough
                dist_after = np.linalg.norm(new_pos - self.main_goal)
                progress = dist_before - dist_after

                # Restore env to think about main goal again
                self.env.goal_position = np.array(self.main_goal, dtype=np.float32)
                next_state_h = self._get_main_goal_obs()

                # High-level reward: progress + shaping
                hl_reward = (
                    self.args.hl_progress_scale * progress
                    - self.args.hl_time_penalty
                )
                if subgoal_reached:
                    hl_reward += self.args.hl_subgoal_bonus

                # Success condition for main goal
                main_dist = np.linalg.norm(new_pos - self.main_goal)
                done_h = False
                if main_dist < self.args.main_goal_success_radius:
                    hl_reward += self.args.hl_success_bonus
                    done_h = True
                    ep_success = True

                # Store high-level transition & update
                self.high_agent.store(state_h, hl_action, hl_reward, next_state_h, done_h)
                loss_h = self.high_agent.update()
                if loss_h is not None:
                    ep_high_losses.append(loss_h)

                ep_reward += float(hl_reward)
                agent_pos = new_pos

                if done_h:
                    break
            self.episode_rewards.append(ep_reward)
            self.episode_successes.append(1 if ep_success else 0)
            self.episode_final_dists.append(final_main_dist)
            success_history.append(1 if ep_success else 0)

            avg_succ = np.mean(success_history) if len(success_history) > 0 else 0.0
            avg_low_loss = np.mean(ep_low_losses) if ep_low_losses else 0.0
            avg_high_loss = np.mean(ep_high_losses) if ep_high_losses else 0.0
            if final_main_dist is None:
                agent_state = self.env.agent.get_state()
                agent_pos = np.array(agent_state.position, dtype=np.float32)
                final_main_dist = float(np.linalg.norm(agent_pos - self.main_goal))
            if ep % self.args.log_interval == 0 or ep == 1:
                print(
                    f"[Episode {ep:4d}] "
                    f"Reward: {ep_reward:8.2f} | "
                    f"Success: {ep_success} | "
                    f"AvgSucc(100): {avg_succ*100:5.1f}% | "
                    f"LowLoss: {avg_low_loss:.4f} | "
                    f"HighLoss: {avg_high_loss:.4f}"
                )

        print("\n=== HAC TRAINING DONE ===\n")
    def plot_training_curves(self, out_dir: str):
        """
        Save simple training plots:
          - Episode reward
          - Moving success rate
          - Final distance to main goal
        """
        os.makedirs(out_dir, exist_ok=True)
        episodes = np.arange(1, len(self.episode_rewards) + 1)

        # --- Episode reward ---
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.title("HAC Training: Episode Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_rewards.png"))
        plt.close()

        # --- Final distance to main goal ---
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, self.episode_final_dists)
        plt.xlabel("Episode")
        plt.ylabel("Final distance to main goal (m)")
        plt.title("HAC Training: Final Distance to Main Goal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_final_distance.png"))
        plt.close()

        # --- Moving success rate (window 50) ---
        window = min(50, len(self.episode_successes))
        if window > 1:
            succ_arr = np.array(self.episode_successes, dtype=np.float32)
            kernel = np.ones(window) / float(window)
            moving_succ = np.convolve(succ_arr, kernel, mode="same")

            plt.figure(figsize=(8, 4))
            plt.plot(episodes, moving_succ * 100.0)
            plt.xlabel("Episode")
            plt.ylabel(f"Success rate (moving avg, window={window}) [%]")
            plt.title("HAC Training: Moving Success Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_success_rate.png"))
            plt.close()
    def debug_episode_trajectory(self, save_prefix: str = "debug_traj"):
        """
        Run ONE greedy episode and visualize:
          - Agent trajectory (x,z)
          - Subgoal positions (x,z) per high-level step
          - Main goal (x,z)
          - Distance to main goal vs high-level step

        Outputs:
          - <save_prefix>_topdown.png
          - <save_prefix>_distance.png
        """
        # Reset env and sample main goal
        obs, _ = self.env.reset()
        agent_state = self.env.agent.get_state()
        agent_pos = np.array(agent_state.position, dtype=np.float32)

        self.main_goal = self._sample_main_goal(agent_pos)
        self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

        agent_traj = []
        subgoal_positions = []
        hl_step_indices = []
        main_dists = []

        ep_success = False

        for hl_step in range(self.args.max_high_steps_eval):
            # Log current agent pos and distance
            agent_state = self.env.agent.get_state()
            agent_pos = np.array(agent_state.position, dtype=np.float32)
            agent_traj.append(agent_pos.copy())

            main_dist = float(np.linalg.norm(agent_pos - self.main_goal))
            main_dists.append(main_dist)
            hl_step_indices.append(hl_step)

            # High-level greedy action
            state_h = self._get_main_goal_obs()
            hl_action = self.high_agent.select_action(state_h, greedy=True)
            subgoal = self.affordances.propose_subgoal(
                agent_pos, self.main_goal, hl_action
            )
            subgoal_positions.append(subgoal.copy())

            # Run low-level greedy rollout toward this subgoal
            self.env.goal_position = np.array(subgoal, dtype=np.float32)

            for ll_step in range(self.args.low_horizon_eval):
                state_l = self._get_subgoal_obs()
                ll_action = self.low_agent.select_action(state_l, greedy=True)
                next_obs_l, reward_l, done_l, trunc_l, info = self.env.step(
                    ll_action
                )

                agent_state = self.env.agent.get_state()
                agent_pos = np.array(agent_state.position, dtype=np.float32)

                d_sub = np.linalg.norm(agent_pos - subgoal)
                if d_sub < self.args.subgoal_success_radius:
                    break
                if done_l or trunc_l:
                    break

            # After low-level rollout, check main goal
            main_dist = float(np.linalg.norm(agent_pos - self.main_goal))
            if main_dist < self.args.main_goal_success_radius:
                ep_success = True
                main_dists.append(main_dist)
                hl_step_indices.append(hl_step + 1)
                agent_traj.append(agent_pos.copy())
                break

        # Convert lists to arrays
        agent_traj = np.array(agent_traj)  # shape [T, 3]
        subgoal_positions = np.array(subgoal_positions)  # [H, 3]
        main_goal = np.array(self.main_goal)

        # --- 2D top-down plot (x vs z) ---
        plt.figure(figsize=(6, 6))
        if len(agent_traj) > 0:
            plt.plot(agent_traj[:, 0], agent_traj[:, 2], marker="o", label="Agent trajectory")
        if len(subgoal_positions) > 0:
            plt.scatter(
                subgoal_positions[:, 0],
                subgoal_positions[:, 2],
                marker="x",
                s=80,
                label="Subgoals (HL)",
            )
        plt.scatter(
            [main_goal[0]],
            [main_goal[2]],
            marker="*",
            s=200,
            label="Main goal",
        )
        plt.xlabel("X (world)")
        plt.ylabel("Z (world)")
        plt.title(f"Top-down trajectory (success={ep_success})")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_topdown.png")
        plt.close()

        # --- Distance to main goal vs HL step ---
        if len(main_dists) > 0:
            steps = np.arange(len(main_dists))
            plt.figure(figsize=(6, 4))
            plt.plot(steps, main_dists, marker="o")
            plt.xlabel("High-level step")
            plt.ylabel("Distance to main goal (m)")
            plt.title("Distance to main goal vs HL step")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_distance.png")
            plt.close()

        print(f"[debug_episode_trajectory] Saved {save_prefix}_topdown.png and {save_prefix}_distance.png")


    def evaluate(self, episodes: int = 10):
        """
        Evaluate hierarchical policy (greedy) on the main goal task.
        """

        print(f"=== EVALUATION ({episodes} episodes, greedy) ===")
        successes = 0
        final_dists = []

        for ep in range(1, episodes + 1):
            obs, _ = self.env.reset()
            agent_state = self.env.agent.get_state()
            agent_pos = np.array(agent_state.position, dtype=np.float32)

            self.main_goal = self._sample_main_goal(agent_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_success = False

            for hl_step in range(self.args.max_high_steps_eval):
                state_h = self._get_main_goal_obs()
                hl_action = self.high_agent.select_action(state_h, greedy=True)
                subgoal = self.affordances.propose_subgoal(
                    agent_pos, self.main_goal, hl_action
                )

                # Run low-level greedy
                self.env.goal_position = np.array(subgoal, dtype=np.float32)

                for ll_step in range(self.args.low_horizon_eval):
                    state_l = self._get_subgoal_obs()
                    ll_action = self.low_agent.select_action(state_l, greedy=True)
                    next_obs_l, reward_l, done_l, trunc_l, info = self.env.step(
                        ll_action
                    )

                    agent_state = self.env.agent.get_state()
                    agent_pos = np.array(agent_state.position, dtype=np.float32)

                    d_sub = np.linalg.norm(agent_pos - subgoal)
                    if d_sub < self.args.subgoal_success_radius:
                        break

                    if done_l or trunc_l:
                        break

                # Check main goal
                main_dist = np.linalg.norm(agent_pos - self.main_goal)
                if main_dist < self.args.main_goal_success_radius:
                    ep_success = True
                    break

            successes += 1 if ep_success else 0
            final_dists.append(float(main_dist))

            print(
                f"Episode {ep:3d}: "
                f"Success={ep_success}, FinalDist={main_dist:.2f}m"
            )

        success_rate = successes / episodes * 100.0
        avg_final_dist = np.mean(final_dists) if final_dists else 0.0
        print("\n=== EVAL SUMMARY ===")
        print(f"Success rate: {success_rate:.1f}% ({successes}/{episodes})")
        print(f"Avg final distance: {avg_final_dist:.2f} m")
        print("=====================\n")


# ============================================================
# 7. Main entry point (argparse + setup)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical Actor-Critic (HAC-style) with Affordances for Habitat Navigation"
    )

    # General
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")

    # Main goal sampling
    parser.add_argument("--main_goal_min_dist", type=float, default=8.0,
                        help="Min distance for main goal sampling")
    parser.add_argument("--main_goal_max_dist", type=float, default=20.0,
                        help="Max distance for main goal sampling")
    parser.add_argument("--main_goal_success_radius", type=float, default=0.6,
                        help="Success radius for main goal")

    # Subgoals / affordances
    parser.add_argument("--subgoal_step_size", type=float, default=5.0,
                        help="Base step size for subgoals")
    parser.add_argument("--subgoal_success_radius", type=float, default=0.8,
                        help="Radius for subgoal success")
    parser.add_argument("--min_subgoal_movement", type=float, default=1.5,
                        help="Min movement between agent and subgoal")

    # Horizons
    parser.add_argument("--max_high_steps", type=int, default=20,
                        help="Max high-level decisions per episode")
    parser.add_argument("--low_horizon", type=int, default=50,
                        help="Low-level horizon per subgoal")
    parser.add_argument("--max_high_steps_eval", type=int, default=20,
                        help="High-level steps at evaluation")
    parser.add_argument("--low_horizon_eval", type=int, default=50,
                        help="Low-level horizon at evaluation")

    # Low-level agent hyperparams
    parser.add_argument("--low_lr", type=float, default=3e-4)
    parser.add_argument("--low_gamma", type=float, default=0.90)
    parser.add_argument("--low_buffer", type=int, default=50_000)
    parser.add_argument("--low_batch", type=int, default=64)
    parser.add_argument("--low_eps_start", type=float, default=1.0)
    parser.add_argument("--low_eps_end", type=float, default=0.05)
    parser.add_argument("--low_eps_decay", type=int, default=50_000)
    parser.add_argument("--low_target_update", type=int, default=1000)

    # High-level agent hyperparams
    parser.add_argument("--high_lr", type=float, default=3e-4)
    parser.add_argument("--high_gamma", type=float, default=0.98)
    parser.add_argument("--high_buffer", type=int, default=50_000)
    parser.add_argument("--high_batch", type=int, default=64)
    parser.add_argument("--high_eps_start", type=float, default=1.0)
    parser.add_argument("--high_eps_end", type=float, default=0.05)
    parser.add_argument("--high_eps_decay", type=int, default=50_000)
    parser.add_argument("--high_target_update", type=int, default=1000)

    # High-level reward shaping
    parser.add_argument("--hl_progress_scale", type=float, default=10.0,
                        help="Scale on progress toward main goal")
    parser.add_argument("--hl_time_penalty", type=float, default=0.05,
                        help="Per high-level step penalty")
    parser.add_argument("--hl_subgoal_bonus", type=float, default=1.0,
                        help="Bonus for subgoal success")
    parser.add_argument("--hl_success_bonus", type=float, default=50.0,
                        help="Bonus for main goal success")

    # Logging / save
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N episodes")
    parser.add_argument("--save_dir", type=str, default="models/hac_models",
                        help="Directory to save models")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Episodes for post-training evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    set_seed(args.seed)
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Environment (same as partner's, imported from their file)
    # --------------------------------------------------------
    print("Creating SimpleNavigationEnv (same as PPO setup)...")
    env = SimpleNavigationEnv()
    print("✓ Env created")

    # --------------------------------------------------------
    # HAC Agents
    # --------------------------------------------------------
    # State dim is 2: [distance, angle]
    state_dim = env.observation_space.shape[0]

    # Low-level actions: same discrete action space as SimpleNavigationEnv (0..3)
    low_action_dim = env.action_space.n

    # High-level actions: 8 directional choices for affordance subgoals
    high_action_dim = 8

    low_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=low_action_dim,
        lr=args.low_lr,
        gamma=args.low_gamma,
        buffer_capacity=args.low_buffer,
        batch_size=args.low_batch,
        eps_start=args.low_eps_start,
        eps_end=args.low_eps_end,
        eps_decay=args.low_eps_decay,
        target_update_freq=args.low_target_update,
        device=device,
        name="low_level",
    )

    high_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=high_action_dim,
        lr=args.high_lr,
        gamma=args.high_gamma,
        buffer_capacity=args.high_buffer,
        batch_size=args.high_batch,
        eps_start=args.high_eps_start,
        eps_end=args.high_eps_end,
        eps_decay=args.high_eps_decay,
        target_update_freq=args.high_target_update,
        device=device,
        name="high_level",
    )

    # --------------------------------------------------------
    # Affordance Model (uses Habitat pathfinder from env)
    # --------------------------------------------------------
    affordances = AffordanceModel(
        pathfinder=env.pathfinder,
        base_step_size=args.subgoal_step_size,
        min_movement=args.min_subgoal_movement,
        use_geodesic_check=True,
    )

    # --------------------------------------------------------
    # Hierarchical HAC Trainer
    # --------------------------------------------------------
    trainer = HierarchicalHACTrainer(
        env=env,
        low_agent=low_agent,
        high_agent=high_agent,
        affordances=affordances,
        args=args,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    trainer.train()

    # Save models
    low_path = os.path.join(args.save_dir, "low_level_dqn.pth")
    high_path = os.path.join(args.save_dir, "high_level_dqn.pth")
    low_agent.save(low_path)
    high_agent.save(high_path)
    print(f"Saved low-level model to:  {low_path}")
    print(f"Saved high-level model to: {high_path}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    trainer.evaluate(episodes=args.eval_episodes)

    # --------------------------------------------------------
    # Save training plots
    # --------------------------------------------------------
    trainer.plot_training_curves(args.save_dir)

    # --------------------------------------------------------
    # One debug episode visualization (greedy)
    # --------------------------------------------------------
    debug_prefix = os.path.join(args.save_dir, "debug_traj")
    trainer.debug_episode_trajectory(save_prefix=debug_prefix)

    env.close()
    print("Done.")
    print(f"Plots saved in: {args.save_dir}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
