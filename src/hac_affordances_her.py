#!/usr/bin/env python3
"""
HAC-style hierarchical navigation with affordances + hindsight

Stage 1 (low level):
  - Train a PPO controller on SimpleNavigationEnv to reach its goal.
  - Save to disk and then freeze it.

Stage 2 (high level):
  - Use frozen PPO as low-level skill.
  - High-level DQN selects discrete subgoal directions.
  - AffordanceModel converts directions -> navigable subgoals.
  - Low-level executes toward subgoal for a fixed horizon.
  - High-level transitions use:
        * normal rewards (progress toward main goal)
        * hindsight relabeling with final position as pseudo-goal.

Produces:
  - Trained low-level PPO model
  - Trained high-level DQN
  - Training curves and one debug trajectory figure.
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

# For low-level PPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# For plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Replay Buffer for DQN
# ============================================================

class ReplayBuffer:
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
# 3. Q-network + DQN agent for high level
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


class DQNAgent:
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
        name: str = "hl_agent",
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
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(
            0.0, (self.eps_decay - self.training_steps) / float(self.eps_decay)
        )
        return eps

    def select_action(self, state: np.ndarray, greedy: bool = False):
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

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + (1.0 - dones) * self.gamma * next_q_target

        loss = nn.functional.mse_loss(q_values, targets)

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
# 4. Affordance model (direction -> subgoal)
# ============================================================

class AffordanceModel:
    """
    Affordance-style subgoal generator.

    High-level actions = 8 compass directions in X-Z plane.
    We bias them slightly toward the main goal and snap to navmesh.
    """

    DIRECTIONS = {
        0: np.array([0.0, 1.0]),    # North (Z+)
        1: np.array([1.0, 1.0]),    # NE
        2: np.array([1.0, 0.0]),    # East
        3: np.array([1.0, -1.0]),   # SE
        4: np.array([0.0, -1.0]),   # South
        5: np.array([-1.0, -1.0]),  # SW
        6: np.array([-1.0, 0.0]),   # West
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
        Convert discrete direction -> navigable 3D subgoal.

        We blend the chosen compass direction with the direction
        pointing to the main goal, then project onto navmesh.
        """
        agent_pos = np.array(agent_pos, dtype=np.float32)
        main_goal = np.array(main_goal, dtype=np.float32)

        base_dir = self._unit_direction(action)  # [dx, dz]

        # Direction to main goal
        g_vec = main_goal - agent_pos
        g_vec[1] = 0.0
        if np.linalg.norm(g_vec) > 1e-6:
            g_vec /= np.linalg.norm(g_vec)
            g_angle = math.atan2(g_vec[2], g_vec[0])
        else:
            g_angle = 0.0

        base_angle = math.atan2(base_dir[1], base_dir[0])
        blended_angle = 0.75 * base_angle + 0.25 * g_angle
        blended_dir = np.array(
            [math.cos(blended_angle), math.sin(blended_angle)], dtype=np.float32
        )

        radii = [
            self.base_step_size,
            self.base_step_size * 0.7,
            self.base_step_size * 0.5,
        ]

        import habitat_sim  # local import

        for r in radii:
            dx, dz = blended_dir[0] * r, blended_dir[1] * r
            raw = np.array(
                [agent_pos[0] + dx, agent_pos[1], agent_pos[2] + dz],
                dtype=np.float32,
            )

            snapped = self.pathfinder.snap_point(raw)
            goal = np.array(snapped, dtype=np.float32)

            if np.isnan(goal).any():
                continue

            movement = np.linalg.norm(goal - agent_pos)
            if movement < self.min_movement:
                continue

            if self.use_geodesic_check:
                path = habitat_sim.ShortestPath()
                path.requested_start = agent_pos
                path.requested_end = goal
                if not self.pathfinder.find_path(path):
                    continue
                if path.geodesic_distance >= 999.0:
                    continue

            return goal

        # Fallback: move a bit toward main goal
        fallback = main_goal - agent_pos
        fallback[1] = 0.0
        if np.linalg.norm(fallback) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            fallback /= np.linalg.norm(fallback)
        candidate = agent_pos + fallback * 2.0
        snapped = self.pathfinder.snap_point(candidate)
        return np.array(snapped, dtype=np.float32)


# ============================================================
# 5. Low-level PPO training (single-stage)
# ============================================================

def train_low_level_ppo(args):
    """
    Train PPO on SimpleNavigationEnv to reach its own goal_position.
    This is basically your partner's low-level script, wrapped.
    """
    print("\n=== LOW-LEVEL PPO TRAINING ===")
    os.makedirs(os.path.dirname(args.low_model_path), exist_ok=True)

    env = SimpleNavigationEnv()
    eval_env = SimpleNavigationEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.90,
        gae_lambda=0.90,
        clip_range=0.2,
        ent_coef=0.005,
        n_epochs=4,
        tensorboard_log=args.low_tensorboard_log,
        verbose=1,
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=args.low_checkpoint_dir,
        name_prefix="lowlevel_nav",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.low_best_dir,
        n_eval_episodes=10,
        eval_freq=20_000,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.low_total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True,
    )

    model.save(args.low_model_path)
    env.close()
    eval_env.close()
    print(f"✓ Low-level PPO saved to: {args.low_model_path}\n")


# ============================================================
# 6. High-level trainer with HER
# ============================================================

class HighLevelTrainerHER:
    """
    High-level DQN with hindsight relabeling on top of a frozen low-level PPO.

    State: [distance_to_main_goal, angle_to_main_goal]
    Action: discrete direction (0..7) -> subgoal via AffordanceModel
    """

    def __init__(
        self,
        env: SimpleNavigationEnv,
        low_model_path: str,
        agent: DQNAgent,
        affordances: AffordanceModel,
        args,
    ):
        self.env = env
        self.low = PPO.load(low_model_path)
        self.low.policy.eval()  # freeze behavior
        self.agent = agent
        self.affordances = affordances
        self.args = args

        self.main_goal = None

        # logs
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_final_dists = []

    # -------- helper: HL obs for arbitrary goal ----------
    def _obs_for_goal(self, agent_pos, agent_rot, goal_pos):
        """
        Construct [distance, angle] observation for a given
        agent pos/rot and goal position, using env internals.
        """
        import habitat_sim

        # Save current state
        current_state = self.env.agent.get_state()

        # Set to provided state
        s = habitat_sim.AgentState()
        s.position = agent_pos
        s.rotation = agent_rot
        self.env.agent.set_state(s)

        self.env.goal_position = np.array(goal_pos, dtype=np.float32)
        obs = self.env._get_obs()
        obs = np.nan_to_num(obs, nan=0.0)

        # Restore current state
        self.env.agent.set_state(current_state)

        return obs

    def _sample_main_goal(self, agent_pos):
        pathfinder = self.env.pathfinder
        min_d = self.args.main_goal_min_dist
        max_d = self.args.main_goal_max_dist
        g = None
        for _ in range(100):
            candidate = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
            d = np.linalg.norm(candidate - agent_pos)
            if min_d <= d <= max_d:
                g = candidate
                break
        if g is None:
            g = candidate
        return g

    def train(self):
        print("\n=== HIGH-LEVEL DQN + HER TRAINING ===\n")
        success_window = deque(maxlen=100)

        import habitat_sim

        for ep in range(1, self.args.episodes + 1):
            obs_ll, _ = self.env.reset()
            agent_state = self.env.agent.get_state()
            agent_pos = np.array(agent_state.position, dtype=np.float32)

            self.main_goal = self._sample_main_goal(agent_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_reward = 0.0
            ep_success = False
            final_main_dist = None
            ep_losses = []

            # store stuff for HER
            her_steps = []

            for hl_step in range(self.args.max_high_steps):
                # HL state w.r.t main goal
                state_h = self._obs_for_goal(
                    np.array(self.env.agent.get_state().position, dtype=np.float32),
                    self.env.agent.get_state().rotation,
                    self.main_goal,
                )

                # select direction
                hl_action = self.agent.select_action(state_h, greedy=False)

                # record agent state before
                state_before = self.env.agent.get_state()
                pos_before = np.array(state_before.position, dtype=np.float32)
                rot_before = state_before.rotation

                # subgoal from affordances
                subgoal = self.affordances.propose_subgoal(
                    pos_before, self.main_goal, hl_action
                )

                # run low-level PPO toward subgoal
                self.env.goal_position = np.array(subgoal, dtype=np.float32)

                for _ in range(self.args.low_horizon):
                    obs_l = self.env._get_obs()
                    obs_l = np.nan_to_num(obs_l, nan=0.0)
                    ll_action, _ = self.low.predict(obs_l, deterministic=True)
                    _, _, done_l, trunc_l, info_l = self.env.step(ll_action)

                    # check subgoal reached
                    cur_pos = np.array(
                        self.env.agent.get_state().position, dtype=np.float32
                    )
                    d_sub = np.linalg.norm(cur_pos - subgoal)
                    if d_sub < self.args.subgoal_success_radius:
                        break
                    if done_l or trunc_l:
                        break

                # state after LL rollout
                state_after = self.env.agent.get_state()
                pos_after = np.array(state_after.position, dtype=np.float32)
                rot_after = state_after.rotation

                # HL progress reward toward main goal
                dist_before = np.linalg.norm(pos_before - self.main_goal)
                dist_after = np.linalg.norm(pos_after - self.main_goal)
                progress = dist_before - dist_after

                self.env.goal_position = np.array(self.main_goal, dtype=np.float32)
                next_state_h = self._obs_for_goal(
                    pos_after, rot_after, self.main_goal
                )

                hl_reward = (
                    self.args.hl_progress_scale * progress
                    - self.args.hl_time_penalty
                )
                done_h = False
                main_dist = dist_after

                if main_dist < self.args.main_goal_success_radius:
                    hl_reward += self.args.hl_success_bonus
                    done_h = True
                    ep_success = True

                self.agent.store(state_h, hl_action, hl_reward, next_state_h, done_h)
                loss = self.agent.update()
                if loss is not None:
                    ep_losses.append(loss)

                ep_reward += float(hl_reward)
                final_main_dist = float(main_dist)

                # store for HER
                her_steps.append(
                    dict(
                        pos_before=pos_before,
                        rot_before=rot_before,
                        pos_after=pos_after,
                        rot_after=rot_after,
                        action=hl_action,
                    )
                )

                if done_h:
                    break

            # --- HER: final position as pseudo-goal ---
            if len(her_steps) > 0:
                final_pos = her_steps[-1]["pos_after"]
                pseudo_goal = final_pos.copy()

                for i, step in enumerate(her_steps):
                    pb = step["pos_before"]
                    rb = step["rot_before"]
                    pa = step["pos_after"]
                    ra = step["rot_after"]
                    act = step["action"]

                    s_her = self._obs_for_goal(pb, rb, pseudo_goal)
                    s_next_her = self._obs_for_goal(pa, ra, pseudo_goal)

                    dist_next = np.linalg.norm(pa - pseudo_goal)
                    if dist_next < self.args.main_goal_success_radius:
                        r_her = self.args.hl_success_bonus
                        d_her = True
                    else:
                        r_her = -self.args.hl_time_penalty
                        d_her = False

                    self.agent.store(s_her, act, r_her, s_next_her, d_her)
                    loss = self.agent.update()
                    if loss is not None:
                        ep_losses.append(loss)

            if final_main_dist is None:
                # fallback: compute from current state
                agent_state = self.env.agent.get_state()
                pos = np.array(agent_state.position, dtype=np.float32)
                final_main_dist = float(np.linalg.norm(pos - self.main_goal))

            self.episode_rewards.append(ep_reward)
            self.episode_successes.append(1 if ep_success else 0)
            self.episode_final_dists.append(final_main_dist)
            success_window.append(1 if ep_success else 0)

            avg_succ = np.mean(success_window) if len(success_window) > 0 else 0.0
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0

            if ep % self.args.log_interval == 0 or ep == 1:
                print(
                    f"[HL Episode {ep:4d}] "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Success: {ep_success} | "
                    f"AvgSucc(100): {avg_succ*100:5.1f}% | "
                    f"AvgLoss: {avg_loss:.4f} | "
                    f"FinalDist: {final_main_dist:5.2f}m"
                )

        print("\n=== HIGH-LEVEL TRAINING DONE ===\n")

    # ---------- plotting & debug -------

    def plot_training_curves(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        episodes = np.arange(1, len(self.episode_rewards) + 1)

        # Reward
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.title("HL Training: Episode Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hl_training_rewards.png"))
        plt.close()

        # Final distance
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, self.episode_final_dists)
        plt.xlabel("Episode")
        plt.ylabel("Final distance to main goal (m)")
        plt.title("HL Training: Final Distance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hl_training_final_distance.png"))
        plt.close()

        # Moving success
        window = min(50, len(self.episode_successes))
        if window > 1:
            succ_arr = np.array(self.episode_successes, dtype=np.float32)
            kernel = np.ones(window) / float(window)
            moving_succ = np.convolve(succ_arr, kernel, mode="same")

            plt.figure(figsize=(8, 4))
            plt.plot(episodes, moving_succ * 100.0)
            plt.xlabel("Episode")
            plt.ylabel(f"Success rate (moving avg, window={window}) [%]")
            plt.title("HL Training: Moving Success Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "hl_training_success_rate.png"))
            plt.close()

    def debug_episode_trajectory(self, save_prefix: str = "hl_debug"):
        """
        One greedy episode with visualizations:
          - top-down XZ trajectory + subgoals + main goal
          - distance to main goal vs HL step
        """
        obs_ll, _ = self.env.reset()
        state = self.env.agent.get_state()
        pos = np.array(state.position, dtype=np.float32)

        self.main_goal = self._sample_main_goal(pos)
        self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

        agent_traj = []
        subgoals = []
        main_dists = []

        for hl_step in range(self.args.max_high_steps_eval):
            state = self.env.agent.get_state()
            pos = np.array(state.position, dtype=np.float32)
            rot = state.rotation

            agent_traj.append(pos.copy())
            d_main = np.linalg.norm(pos - self.main_goal)
            main_dists.append(d_main)

            s_h = self._obs_for_goal(pos, rot, self.main_goal)
            a = self.agent.select_action(s_h, greedy=True)

            subgoal = self.affordances.propose_subgoal(pos, self.main_goal, a)
            subgoals.append(subgoal.copy())

            # run low level
            self.env.goal_position = np.array(subgoal, dtype=np.float32)
            for _ in range(self.args.low_horizon_eval):
                obs_l = self.env._get_obs()
                obs_l = np.nan_to_num(obs_l, nan=0.0)
                ll_action, _ = self.low.predict(obs_l, deterministic=True)
                _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                pos = np.array(
                    self.env.agent.get_state().position, dtype=np.float32
                )
                if np.linalg.norm(pos - subgoal) < self.args.subgoal_success_radius:
                    break
                if done_l or trunc_l:
                    break

            if np.linalg.norm(pos - self.main_goal) < self.args.main_goal_success_radius:
                agent_traj.append(pos.copy())
                main_dists.append(np.linalg.norm(pos - self.main_goal))
                break

        agent_traj = np.array(agent_traj)
        subgoals = np.array(subgoals)
        main_goal = np.array(self.main_goal)

        # top-down
        plt.figure(figsize=(6, 6))
        if len(agent_traj) > 0:
            plt.plot(agent_traj[:, 0], agent_traj[:, 2],
                     marker="o", label="Agent")
        if len(subgoals) > 0:
            plt.scatter(subgoals[:, 0], subgoals[:, 2],
                        marker="x", s=80, label="Subgoals")
        plt.scatter([main_goal[0]], [main_goal[2]],
                    marker="*", s=200, label="Main goal")
        plt.xlabel("X (world)")
        plt.ylabel("Z (world)")
        plt.title("HL debug: top-down")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_topdown.png")
        plt.close()

        # distance vs HL step
        if len(main_dists) > 0:
            steps = np.arange(len(main_dists))
            plt.figure(figsize=(6, 4))
            plt.plot(steps, main_dists, marker="o")
            plt.xlabel("HL step")
            plt.ylabel("Distance to main goal (m)")
            plt.title("HL debug: distance vs HL step")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_distance.png")
            plt.close()

        print(f"[debug] saved {save_prefix}_topdown.png and {save_prefix}_distance.png")


# ============================================================
# 7. Argparse + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HAC-style hierarchical navigation with affordances + HER"
    )

    # general
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=500,
                   help="HL training episodes")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="models/hac_her_models")

    # low-level PPO
    p.add_argument("--low_model_path", type=str,
                   default="models/lowlevel_ppo")
    p.add_argument("--low_total_timesteps", type=int, default=250_000)
    p.add_argument("--skip_low_train", action="store_true",
                   help="Skip low-level training if model exists")
    p.add_argument("--low_tensorboard_log", type=str,
                   default="./logs_lowlevel/")
    p.add_argument("--low_checkpoint_dir", type=str,
                   default="./models/lowlevel_checkpoints/")
    p.add_argument("--low_best_dir", type=str,
                   default="./models/lowlevel_best/")

    # main goal
    p.add_argument("--main_goal_min_dist", type=float, default=8.0)
    p.add_argument("--main_goal_max_dist", type=float, default=20.0)
    p.add_argument("--main_goal_success_radius", type=float, default=0.6)

    # subgoals
    p.add_argument("--subgoal_step_size", type=float, default=5.0)
    p.add_argument("--subgoal_success_radius", type=float, default=0.8)
    p.add_argument("--min_subgoal_movement", type=float, default=1.5)

    # horizons
    p.add_argument("--max_high_steps", type=int, default=20)
    p.add_argument("--low_horizon", type=int, default=50)
    p.add_argument("--max_high_steps_eval", type=int, default=20)
    p.add_argument("--low_horizon_eval", type=int, default=50)

    # high-level DQN hyperparams
    p.add_argument("--hl_lr", type=float, default=3e-4)
    p.add_argument("--hl_gamma", type=float, default=0.98)
    p.add_argument("--hl_buffer", type=int, default=50_000)
    p.add_argument("--hl_batch", type=int, default=64)
    p.add_argument("--hl_eps_start", type=float, default=1.0)
    p.add_argument("--hl_eps_end", type=float, default=0.05)
    p.add_argument("--hl_eps_decay", type=int, default=50_000)
    p.add_argument("--hl_target_update", type=int, default=1000)

    # high-level reward shaping
    p.add_argument("--hl_progress_scale", type=float, default=10.0)
    p.add_argument("--hl_time_penalty", type=float, default=0.05)
    p.add_argument("--hl_success_bonus", type=float, default=50.0)

    p.add_argument("--eval_episodes", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    args.device = device
    set_seed(args.seed)
    print(f"Using device: {device}")

    # -------- Stage 1: low-level PPO --------
    if args.skip_low_train and os.path.exists(args.low_model_path + ".zip"):
        print(f"Skipping low-level training, using existing {args.low_model_path}.zip")
    else:
        train_low_level_ppo(args)

    low_model_path = args.low_model_path

    # -------- Stage 2: high-level DQN + HER --------
    env = SimpleNavigationEnv()

    state_dim = env.observation_space.shape[0]  # [distance, angle] = 2
    hl_action_dim = 8  # 8 compass directions

    hl_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=hl_action_dim,
        lr=args.hl_lr,
        gamma=args.hl_gamma,
        buffer_capacity=args.hl_buffer,
        batch_size=args.hl_batch,
        eps_start=args.hl_eps_start,
        eps_end=args.hl_eps_end,
        eps_decay=args.hl_eps_decay,
        target_update_freq=args.hl_target_update,
        device=device,
        name="high_level",
    )

    affordances = AffordanceModel(
        pathfinder=env.pathfinder,
        base_step_size=args.subgoal_step_size,
        min_movement=args.min_subgoal_movement,
        use_geodesic_check=True,
    )

    trainer = HighLevelTrainerHER(
        env=env,
        low_model_path=low_model_path,
        agent=hl_agent,
        affordances=affordances,
        args=args,
    )

    trainer.train()

    # Save HL model
    hl_path = os.path.join(args.save_dir, "high_level_dqn.pth")
    hl_agent.save(hl_path)
    print(f"High-level model saved to: {hl_path}")

    # Plots + debug
    trainer.plot_training_curves(args.save_dir)
    debug_prefix = os.path.join(args.save_dir, "hl_debug")
    trainer.debug_episode_trajectory(save_prefix=debug_prefix)

    env.close()
    print("Done. Outputs saved in:", args.save_dir)


if __name__ == "__main__":
    main()
