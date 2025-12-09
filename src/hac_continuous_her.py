#!/usr/bin/env python3
"""
Hierarchical navigation with:

  - Low-level: PPO controller on SimpleNavigationEnv (frozen during HL training)
  - High-level: continuous actor-critic (DDPG-style) over 2D subgoal offsets
  - Subgoals: generated along pathfinder geodesic, then offset by HL action
  - High-level state: [dist_to_goal, angle_to_goal, agent_x, agent_z, goal_x, goal_z]
  - Hindsight Experience Replay (HER) at the high level

Drop this file in the same directory as simple_navigation_env.py and run:

    python hac_continuous_her.py --episodes 500

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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. General utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Replay buffer for continuous high-level agent
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
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
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_states[i] = next_state
        self.dones[i, 0] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        assert len(self) >= batch_size
        idxs = np.random.randint(0, len(self), size=batch_size)

        s = torch.from_numpy(self.states[idxs]).to(self.device)
        a = torch.from_numpy(self.actions[idxs]).to(self.device)
        r = torch.from_numpy(self.rewards[idxs]).to(self.device)
        s2 = torch.from_numpy(self.next_states[idxs]).to(self.device)
        d = torch.from_numpy(self.dones[idxs]).to(self.device)

        return s, a, r, s2, d


# ============================================================
# 3. Continuous actor-critic (DDPG-style) for high level
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DDPGAgent:
    """
    Simple DDPG-style continuous actor-critic for high-level subgoal selection.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        buffer_capacity: int,
        batch_size: int,
        device: torch.device,
        init_noise_std: float = 0.3,
        min_noise_std: float = 0.05,
        noise_decay_episodes: int = 500,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, device)

        self.total_updates = 0

        self.init_noise_std = init_noise_std
        self.min_noise_std = min_noise_std
        self.noise_decay_episodes = noise_decay_episodes
        self.current_episode = 0

    def set_episode(self, ep_idx: int):
        self.current_episode = ep_idx

    def _noise_std(self):
        frac = max(0.0, 1.0 - self.current_episode / float(self.noise_decay_episodes))
        return self.min_noise_std + (self.init_noise_std - self.min_noise_std) * frac

    def select_action(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(s).cpu().numpy()[0]  # in [-1,1]
        if not greedy:
            noise = np.random.normal(0.0, self._noise_std(), size=self.action_dim)
            a = a + noise
        a = np.clip(a, -1.0, 1.0)
        return a.astype(np.float32)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def soft_update(self, target: nn.Module, source: nn.Module):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def update(self, updates_per_step: int = 1):
        if len(self.buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0

        for _ in range(updates_per_step):
            s, a, r, s2, d = self.buffer.sample(self.batch_size)

            # Critic update
            with torch.no_grad():
                a2 = self.target_actor(s2)
                q2 = self.target_critic(s2, a2)
                target_q = r + (1.0 - d) * self.gamma * q2

            q = self.critic(s, a)
            critic_loss = nn.functional.mse_loss(q, target_q)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            # Actor update (maximize Q => minimize -Q)
            actor_actions = self.actor(s)
            actor_loss = -self.critic(s, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

            total_loss += critic_loss.item()
            self.total_updates += 1

        return total_loss / float(updates_per_step)


# ============================================================
# 4. Geodesic subgoal generator (affordances)
# ============================================================

class GeodesicAffordanceModel:
    """
    Use Habitat pathfinder to compute geodesic path from agent to main goal.
    HL action is a 2D vector in [-1,1]^2 that offsets a waypoint along that path.
    """

    def __init__(
        self,
        pathfinder,
        base_step: float = 3.0,
        offset_scale: float = 1.0,
        min_movement: float = 1.0,
    ):
        self.pathfinder = pathfinder
        self.base_step = float(base_step)
        self.offset_scale = float(offset_scale)
        self.min_movement = float(min_movement)

    def make_subgoal(self, agent_pos: np.ndarray, main_goal: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        import habitat_sim

        agent_pos = np.array(agent_pos, dtype=np.float32)
        main_goal = np.array(main_goal, dtype=np.float32)

        # 1) Compute shortest path
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = main_goal

        ok = self.pathfinder.find_path(path)
        if not ok or len(path.points) == 0:
            # Fallback: straight towards goal
            direction = main_goal - agent_pos
            direction[1] = 0.0
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                direction /= np.linalg.norm(direction)
            base_wp = agent_pos + direction * self.base_step
        else:
            pts = np.array(path.points, dtype=np.float32)
            # march along path until we hit base_step
            dist_acc = 0.0
            base_wp = pts[-1]
            for i in range(1, len(pts)):
                seg = pts[i] - pts[i - 1]
                seglen = float(np.linalg.norm(seg))
                if dist_acc + seglen >= self.base_step:
                    t = (self.base_step - dist_acc) / max(seglen, 1e-6)
                    base_wp = pts[i - 1] + t * seg
                    break
                dist_acc += seglen

        # 2) Apply HL action as local offset in x-z
        ax, az = float(action_vec[0]), float(action_vec[1])
        offset = np.array([ax * self.offset_scale, 0.0, az * self.offset_scale], dtype=np.float32)
        candidate = base_wp + offset

        snapped = self.pathfinder.snap_point(candidate)
        subgoal = np.array(snapped, dtype=np.float32)

        if np.isnan(subgoal).any():
            subgoal = base_wp

        if np.linalg.norm(subgoal - agent_pos) < self.min_movement:
            subgoal = base_wp

        return subgoal


# ============================================================
# 5. Low-level PPO training (same as partner, wrapped)
# ============================================================

def train_low_level_ppo(args):
    """
    Train PPO on SimpleNavigationEnv to reach its own goal_position.
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
    print(f"✓ Low-level PPO saved to: {args.low_model_path}.zip\n")


# ============================================================
# 6. High-level trainer with DDPG + HER + rich state
# ============================================================

class HighLevelDDPGHERTrainer:
    """
    High-level continuous actor-critic over 2D subgoal offsets, with HER.

    State = [dist_to_main_goal, angle_to_main_goal, agent_x, agent_z, goal_x, goal_z]
    """

    def __init__(
        self,
        env: SimpleNavigationEnv,
        low_model_path: str,
        hl_agent: DDPGAgent,
        afford: GeodesicAffordanceModel,
        args,
    ):
        self.env = env
        self.low = PPO.load(low_model_path)
        self.low.policy.eval()  # freeze weights
        self.agent = hl_agent
        self.afford = afford
        self.args = args

        self.main_goal = None

        # logging
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_final_dists = []

    # ---- HL state construction ----

    def _get_hl_state_for_goal(self, agent_pos, agent_rot, goal_pos):
        """
        Build high-level state [dist, angle, agent_x, agent_z, goal_x, goal_z]
        for arbitrary agent pose and goal.
        """
        import habitat_sim

        # Save current state
        cur_state = self.env.agent.get_state()

        s = habitat_sim.AgentState()
        s.position = agent_pos
        s.rotation = agent_rot
        self.env.agent.set_state(s)

        self.env.goal_position = np.array(goal_pos, dtype=np.float32)
        obs = self.env._get_obs()
        obs = np.nan_to_num(obs, nan=0.0)

        dist, angle = float(obs[0]), float(obs[1])
        agent_xz = np.array(agent_pos, dtype=np.float32)[[0, 2]]
        goal_xz = np.array(goal_pos, dtype=np.float32)[[0, 2]]

        hl_state = np.array(
            [dist, angle, agent_xz[0], agent_xz[1], goal_xz[0], goal_xz[1]],
            dtype=np.float32,
        )

        # restore
        self.env.agent.set_state(cur_state)

        return hl_state

    def _get_current_hl_state(self):
        a_state = self.env.agent.get_state()
        return self._get_hl_state_for_goal(
            np.array(a_state.position, dtype=np.float32),
            a_state.rotation,
            self.main_goal,
        )

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

    # ---- Training loop ----

    def train(self):
        print("\n=== HIGH-LEVEL CONTINUOUS DDPG + HER TRAINING ===\n")
        success_window = deque(maxlen=100)

        for ep in range(1, self.args.episodes + 1):
            self.agent.set_episode(ep)

            obs_ll, _ = self.env.reset()
            a_state = self.env.agent.get_state()
            a_pos = np.array(a_state.position, dtype=np.float32)

            self.main_goal = self._sample_main_goal(a_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_reward = 0.0
            ep_success = False
            final_main_dist = None
            ep_losses = []

            # for HER
            her_steps = []

            for hl_step in range(self.args.max_high_steps):
                # state before HL action
                a_state = self.env.agent.get_state()
                pos_before = np.array(a_state.position, dtype=np.float32)
                rot_before = a_state.rotation

                s_h = self._get_hl_state_for_goal(pos_before, rot_before, self.main_goal)

                # continuous HL action
                action = self.agent.select_action(s_h, greedy=False)

                # subgoal from geodesic + offset
                subgoal = self.afford.make_subgoal(pos_before, self.main_goal, action)

                # low-level rollout toward subgoal
                self.env.goal_position = np.array(subgoal, dtype=np.float32)
                for _ in range(self.args.low_horizon):
                    obs_l = self.env._get_obs()
                    obs_l = np.nan_to_num(obs_l, nan=0.0)
                    ll_action, _ = self.low.predict(obs_l, deterministic=True)
                    _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                    cur_pos = np.array(self.env.agent.get_state().position, dtype=np.float32)
                    d_sub = np.linalg.norm(cur_pos - subgoal)
                    if d_sub < self.args.subgoal_success_radius:
                        break
                    if done_l or trunc_l:
                        break

                # state after rollout
                a_state_after = self.env.agent.get_state()
                pos_after = np.array(a_state_after.position, dtype=np.float32)
                rot_after = a_state_after.rotation

                # progress & reward w.r.t main goal
                dist_before = np.linalg.norm(pos_before - self.main_goal)
                dist_after = np.linalg.norm(pos_after - self.main_goal)
                progress = dist_before - dist_after

                self.env.goal_position = np.array(self.main_goal, dtype=np.float32)
                s_h_next = self._get_hl_state_for_goal(pos_after, rot_after, self.main_goal)

                reward = self.args.hl_progress_scale * progress - self.args.hl_time_penalty
                done_h = False

                if dist_after < self.args.main_goal_success_radius:
                    reward += self.args.hl_success_bonus
                    done_h = True
                    ep_success = True

                self.agent.store(s_h, action, reward, s_h_next, done_h)
                loss = self.agent.update(self.args.hl_updates_per_step)
                if loss is not None:
                    ep_losses.append(loss)

                ep_reward += float(reward)
                final_main_dist = float(dist_after)

                # store data for HER
                her_steps.append(
                    dict(
                        pos_before=pos_before,
                        rot_before=rot_before,
                        pos_after=pos_after,
                        rot_after=rot_after,
                        action=action.copy(),
                    )
                )

                if done_h:
                    break

            # HER: use final achieved position as pseudo-goal
            if len(her_steps) > 0:
                final_pos = her_steps[-1]["pos_after"]
                pseudo_goal = final_pos.copy()

                for step in her_steps:
                    pb = step["pos_before"]
                    rb = step["rot_before"]
                    pa = step["pos_after"]
                    ra = step["rot_after"]
                    act = step["action"]

                    s_her = self._get_hl_state_for_goal(pb, rb, pseudo_goal)
                    s_next_her = self._get_hl_state_for_goal(pa, ra, pseudo_goal)

                    dist_next = np.linalg.norm(pa - pseudo_goal)
                    if dist_next < self.args.main_goal_success_radius:
                        r_her = self.args.hl_success_bonus
                        d_her = True
                    else:
                        r_her = -self.args.hl_time_penalty
                        d_her = False

                    self.agent.store(s_her, act, r_her, s_next_her, d_her)
                    loss = self.agent.update(self.args.hl_updates_per_step)
                    if loss is not None:
                        ep_losses.append(loss)

            if final_main_dist is None:
                a_state = self.env.agent.get_state()
                pos = np.array(a_state.position, dtype=np.float32)
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
                    f"AvgCriticLoss: {avg_loss:.4f} | "
                    f"FinalDist: {final_main_dist:5.2f}m"
                )

        print("\n=== HIGH-LEVEL TRAINING DONE ===\n")

    # ---- plotting & debug ----

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
        One greedy HL episode with:
          - top-down X/Z plot (agent, subgoals, main goal)
          - distance-to-goal vs HL step
        """
        obs_ll, _ = self.env.reset()
        a_state = self.env.agent.get_state()
        a_pos = np.array(a_state.position, dtype=np.float32)

        self.main_goal = self._sample_main_goal(a_pos)
        self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

        agent_traj = []
        subgoals = []
        main_dists = []

        for hl_step in range(self.args.max_high_steps_eval):
            a_state = self.env.agent.get_state()
            pos = np.array(a_state.position, dtype=np.float32)
            rot = a_state.rotation

            agent_traj.append(pos.copy())
            main_dists.append(np.linalg.norm(pos - self.main_goal))

            s_h = self._get_hl_state_for_goal(pos, rot, self.main_goal)
            action = self.agent.select_action(s_h, greedy=True)

            subgoal = self.afford.make_subgoal(pos, self.main_goal, action)
            subgoals.append(subgoal.copy())

            self.env.goal_position = np.array(subgoal, dtype=np.float32)
            for _ in range(self.args.low_horizon_eval):
                obs_l = self.env._get_obs()
                obs_l = np.nan_to_num(obs_l, nan=0.0)
                ll_action, _ = self.low.predict(obs_l, deterministic=True)
                _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                pos = np.array(self.env.agent.get_state().position, dtype=np.float32)
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
            plt.plot(agent_traj[:, 0], agent_traj[:, 2], marker="o", label="Agent")
        if len(subgoals) > 0:
            plt.scatter(subgoals[:, 0], subgoals[:, 2], marker="x", s=80, label="Subgoals")
        plt.scatter([main_goal[0]], [main_goal[2]], marker="*", s=200, label="Main goal")
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

        print(f"[debug] Saved {save_prefix}_topdown.png and {save_prefix}_distance.png")

    def evaluate(self, episodes: int = 10):
        """
        Quick greedy evaluation of HL policy.
        """
        print(f"=== HL GREEDY EVALUATION ({episodes} episodes) ===")
        successes = 0
        final_dists = []

        for ep in range(1, episodes + 1):
            obs_ll, _ = self.env.reset()
            a_state = self.env.agent.get_state()
            a_pos = np.array(a_state.position, dtype=np.float32)

            self.main_goal = self._sample_main_goal(a_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_success = False

            for hl_step in range(self.args.max_high_steps_eval):
                a_state = self.env.agent.get_state()
                pos = np.array(a_state.position, dtype=np.float32)
                rot = a_state.rotation

                s_h = self._get_hl_state_for_goal(pos, rot, self.main_goal)
                action = self.agent.select_action(s_h, greedy=True)

                subgoal = self.afford.make_subgoal(pos, self.main_goal, action)
                self.env.goal_position = np.array(subgoal, dtype=np.float32)

                for _ in range(self.args.low_horizon_eval):
                    obs_l = self.env._get_obs()
                    obs_l = np.nan_to_num(obs_l, nan=0.0)
                    ll_action, _ = self.low.predict(obs_l, deterministic=True)
                    _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                    pos = np.array(self.env.agent.get_state().position, dtype=np.float32)
                    if np.linalg.norm(pos - subgoal) < self.args.subgoal_success_radius:
                        break
                    if done_l or trunc_l:
                        break

                if np.linalg.norm(pos - self.main_goal) < self.args.main_goal_success_radius:
                    ep_success = True
                    break

            final_dist = float(np.linalg.norm(pos - self.main_goal))
            successes += 1 if ep_success else 0
            final_dists.append(final_dist)
            print(f"Episode {ep:3d}: success={ep_success}, final_dist={final_dist:.2f}m")

        sr = successes / episodes * 100.0
        avg_dist = float(np.mean(final_dists)) if final_dists else 0.0
        print(f"\nEval success rate: {sr:.1f}% ({successes}/{episodes})")
        print(f"Avg final distance: {avg_dist:.2f}m\n")


# ============================================================
# 7. Argparse + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Continuous HAC-style navigation with geodesic affordances + HER"
    )

    # general
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=500,
                   help="HL training episodes")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="models/hac_continuous_her_models")

    # low-level PPO
    p.add_argument("--low_model_path", type=str,
                   default="models/lowlevel_ppo")
    p.add_argument("--low_total_timesteps", type=int, default=250_000)
    p.add_argument("--skip_low_train", action="store_true",
                   help="Skip low-level PPO training if model exists")
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
    p.add_argument("--subgoal_base_step", type=float, default=3.0)
    p.add_argument("--subgoal_offset_scale", type=float, default=1.0)
    p.add_argument("--subgoal_success_radius", type=float, default=0.8)
    p.add_argument("--min_subgoal_movement", type=float, default=1.0)

    # horizons
    p.add_argument("--max_high_steps", type=int, default=20)
    p.add_argument("--low_horizon", type=int, default=50)
    p.add_argument("--max_high_steps_eval", type=int, default=20)
    p.add_argument("--low_horizon_eval", type=int, default=50)

    # HL DDPG hyperparams
    p.add_argument("--hl_actor_lr", type=float, default=1e-3)
    p.add_argument("--hl_critic_lr", type=float, default=1e-3)
    p.add_argument("--hl_gamma", type=float, default=0.99)
    p.add_argument("--hl_tau", type=float, default=0.005)
    p.add_argument("--hl_buffer", type=int, default=100_000)
    p.add_argument("--hl_batch", type=int, default=128)
    p.add_argument("--hl_init_noise_std", type=float, default=0.3)
    p.add_argument("--hl_min_noise_std", type=float, default=0.05)
    p.add_argument("--hl_noise_decay_episodes", type=int, default=500)
    p.add_argument("--hl_updates_per_step", type=int, default=1)

    # HL reward shaping
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

    # ---- Stage 1: low-level PPO ----
    low_model_zip = args.low_model_path + ".zip"
    if args.skip_low_train and os.path.exists(low_model_zip):
        print(f"Skipping low-level training, found {low_model_zip}")
    else:
        train_low_level_ppo(args)

    # ---- Stage 2: high-level DDPG + HER ----
    env = SimpleNavigationEnv()

    # HL state = [dist, angle, agent_x, agent_z, goal_x, goal_z] -> dim=6
    state_dim = 6
    action_dim = 2  # 2D offset in x-z

    hl_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=args.hl_actor_lr,
        critic_lr=args.hl_critic_lr,
        gamma=args.hl_gamma,
        tau=args.hl_tau,
        buffer_capacity=args.hl_buffer,
        batch_size=args.hl_batch,
        device=device,
        init_noise_std=args.hl_init_noise_std,
        min_noise_std=args.hl_min_noise_std,
        noise_decay_episodes=args.hl_noise_decay_episodes,
    )

    afford = GeodesicAffordanceModel(
        pathfinder=env.pathfinder,
        base_step=args.subgoal_base_step,
        offset_scale=args.subgoal_offset_scale,
        min_movement=args.min_subgoal_movement,
    )

    trainer = HighLevelDDPGHERTrainer(
        env=env,
        low_model_path=args.low_model_path,
        hl_agent=hl_agent,
        afford=afford,
        args=args,
    )

    trainer.train()

    # save HL weights
    hl_actor_path = os.path.join(args.save_dir, "hl_actor.pth")
    hl_critic_path = os.path.join(args.save_dir, "hl_critic.pth")
    torch.save(hl_agent.actor.state_dict(), hl_actor_path)
    torch.save(hl_agent.critic.state_dict(), hl_critic_path)
    print(f"Saved HL actor to  {hl_actor_path}")
    print(f"Saved HL critic to {hl_critic_path}")

    trainer.plot_training_curves(args.save_dir)
    debug_prefix = os.path.join(args.save_dir, "hl_debug")
    trainer.debug_episode_trajectory(save_prefix=debug_prefix)

    trainer.evaluate(args.eval_episodes)

    env.close()
    print("Done. Outputs saved in:", args.save_dir)


if __name__ == "__main__":
    main()
