#!/usr/bin/env python3
"""
Hierarchical navigation with:

  - Low-level: PPO controller on SimpleNavigationEnv (frozen during HL training)
  - High-level: continuous actor-critic (TD3-style) over 2D subgoal offsets
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

from habitat_arm_reaching_env import HabitatArmReachingEnv

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime


import sys
# Map the new 'numpy._core' to the old 'numpy.core' so the model loads
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = np.core.numeric


# ============================================================
# 1. General utils
# ============================================================

# def get_ee_pos(angles):
#     """
#     Compute End-Effector position from joint angles.
#     Must match SimpleArmReachingEnv._get_observation logic.
#     """
#     # ee_pos = np.sum(np.sin(self.arm_angles[:3])) * np.array([1, 1, 1])
#     return np.sum(np.sin(angles[:3])) * np.array([1, 1, 1], dtype=np.float32)

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
# 3. Continuous actor-critic (TD3-style) for high level
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, x):
        return self.net(x) * self.max_action


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


class TD3Agent:
    """
    TD3-style continuous actor-critic for high-level subgoal selection.
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
        max_action: float = 1.0,
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
        self.policy_delay = 2
        self.max_action = max_action
        # Scale noise clip by max_action to allow sufficient exploration
        self.noise_clip = max_action * 0.5

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_opt_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

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
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(state_t)
            if not greedy:
                noise = torch.randn_like(a) * self._noise_std()
                # Clamp noise to allow exploration but prevent instability
                a = a + torch.clamp(noise, -self.noise_clip, self.noise_clip)
            a = torch.clamp(a, -self.max_action, self.max_action)
        return a.cpu().numpy()[0]

    def select_target_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = self.target_actor(state)
            noise = torch.randn_like(a) * self._noise_std()
            a = a + torch.clamp(noise, -self.noise_clip, self.noise_clip)
            a = torch.clamp(a, -self.max_action, self.max_action)
        return a

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def soft_update(self, target: nn.Module, source: nn.Module):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def update(self, updates_per_step: int = 2):
        if len(self.buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0

        for j in range(updates_per_step):
            s, a, r, s2, d = self.buffer.sample(self.batch_size)

            # Critic update
            with torch.no_grad():
                # a2 = self.target_actor(s2)
                a2 = self.select_target_action(s2)
                q21 = self.target_critic_1(s2, a2)
                q22 = self.target_critic_2(s2, a2)
                target_q = r + (1.0 - d) * self.gamma * torch.min(q21, q22)

            q1 = self.critic_1(s, a)
            q2 = self.critic_2(s, a)

            critic_1_loss = nn.functional.mse_loss(q1, target_q)
            critic_2_loss = nn.functional.mse_loss(q2, target_q)
            critic_loss = critic_1_loss + critic_2_loss

            self.critic_opt_1.zero_grad()
            self.critic_opt_2.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
            self.critic_opt_1.step()
            self.critic_opt_2.step()

            # Actor update (maximize Q => minimize -Q)
            if j % self.policy_delay == 0:
                actor_actions = self.actor(s)
                actor_loss = -self.critic_1(s, actor_actions).mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                self.soft_update(self.target_actor, self.actor)
                self.soft_update(self.target_critic_1, self.critic_1)
                self.soft_update(self.target_critic_2, self.critic_2)

                total_loss += critic_1_loss.item() + critic_2_loss.item()
            
            self.total_updates += 1

        return total_loss / float(updates_per_step)


# ============================================================
# 5. Low-level PPO training (same as partner, wrapped)
# ============================================================

def train_low_level_ppo(args):
    """
    Train PPO on SimpleArmReachingEnv to reach its own goal_position.
    """
    print("\n=== LOW-LEVEL PPO TRAINING ===")
    os.makedirs(os.path.dirname(args.low_model_path), exist_ok=True)

    env = HabitatArmReachingEnv(max_steps=200)
    eval_env = HabitatArmReachingEnv(max_steps=200)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        tensorboard_log=args.low_tensorboard_log,
        verbose=1,
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=args.low_checkpoint_dir,
        name_prefix="lowlevel_arm",
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
# 6. High-level trainer with TD3 + HER + rich state
# ============================================================

class HighLevelTD3HERTrainer:
    """
    High-level continuous actor-critic over 3D subgoal offsets, with HER.

    State = [agent_x, agent_y, agent_z, goal_x, goal_y, goal_z, delta_x, delta_y, delta_z]
    """

    def __init__(
        self,
        env: HabitatArmReachingEnv,
        ll_agent,
        hl_agent: TD3Agent,
        args,
    ):
        self.env = env
        self.low = ll_agent
        self.low.policy.eval()  # freeze weights
        self.agent = hl_agent
        self.args = args

        self.main_goal = None

        # logging
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_final_dists = []

    # ---- HL state construction ----

    def _get_hl_state_for_goal(self, agent_pos, goal_pos):
        """
        Build high-level state [agent_x, agent_y, agent_z, goal_x, goal_y, goal_z, delta_x, delta_y, delta_z]
        """
        agent_pos = np.array(agent_pos, dtype=np.float32)
        goal_pos = np.array(goal_pos, dtype=np.float32)
        delta = goal_pos - agent_pos

        hl_state = np.concatenate([agent_pos, goal_pos, delta]).astype(np.float32)
        return hl_state

    def _get_current_hl_state(self):
        agent_pos = self.get_ee_pos(self.env.arm_angles)
        return self._get_hl_state_for_goal(agent_pos, self.main_goal)

    def _sample_main_goal(self, agent_pos):
        # Sample random goal in 3D space, similar to env init
        return np.random.randn(3).astype(np.float32) * 0.5
    
    def _low_level_policy(self, obs, goal):
        # Use SAC to generate actions toward the subgoal
        # obs[0] = distance to goal, obs[1:4] = EE position, obs[4:7] = goal position
        current_pos = obs[1:4]
        goal_pos = obs[4:7]
        
        # Direction toward goal (normalized)
        direction = goal_pos - current_pos
        dist = np.linalg.norm(direction)
        
        if dist > 1e-6:
            direction = direction / dist
        else:
            direction = np.zeros(3)
        
        # SAC also suggests a direction
        sac_action, _ = self.low.predict(obs, deterministic=True)
        
        # Blend: 70% toward goal, 30% SAC suggestion
        # This helps the policy learn while still making progress
        blended = 0.7 * direction + 0.3 * sac_action
        
        # Scale for meaningful movement (0.3m steps)
        action = blended * self.args.low_action_scale
        action = np.clip(action, -self.args.low_action_scale, self.args.low_action_scale)
        
        return action

    def get_ee_pos(self, angles):
        return self.env._forward_kinematics(angles)

    # ---- Training loop ----

    def train(self):
        print("\n=== HIGH-LEVEL CONTINUOUS TD3 + HER TRAINING ===\n")
        success_window = deque(maxlen=100)

        for ep in range(1, self.args.episodes + 1):
            self.agent.set_episode(ep)

            obs_ll, _ = self.env.reset()
            a_pos = self.get_ee_pos(self.env.arm_angles)

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
                pos_before = self.get_ee_pos(self.env.arm_angles)

                s_h = self._get_hl_state_for_goal(pos_before, self.main_goal)

                # continuous HL action (3D offset)
                action = self.agent.select_action(s_h, greedy=False)

                # subgoal = current_pos + action
                subgoal = pos_before + action

                # low-level rollout toward subgoal
                self.env.goal_position = np.array(subgoal, dtype=np.float32)
                
                # Get observation relative to new subgoal
                obs_l = self.env._get_observation()
                
                for _ in range(self.args.low_horizon):
                    ll_action = self._low_level_policy(obs_l, subgoal)
                    
                    obs_l, _, done_l, trunc_l, _ = self.env.step(ll_action)

                    cur_pos = obs_l[1]
                    d_sub = np.linalg.norm(cur_pos - subgoal)
                    if d_sub < self.args.subgoal_success_radius:
                        break
                    if done_l or trunc_l:
                        break

                # state after rollout
                pos_after = self.get_ee_pos(self.env.arm_angles)

                # progress & reward w.r.t main goal
                dist_before = np.linalg.norm(pos_before - self.main_goal)
                dist_after = np.linalg.norm(pos_after - self.main_goal)
                progress = dist_before - dist_after

                self.env.goal_position = np.array(self.main_goal, dtype=np.float32)
                s_h_next = self._get_hl_state_for_goal(pos_after, self.main_goal)

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
                        pos_after=pos_after,
                        action=action.copy(),
                    )
                )

                if done_h:
                    break

            # HER: Future Strategy (k=4)
            # For each transition, sample k goals from the future of the trajectory
            k_future = 4
            if len(her_steps) > 0:
                for t, step in enumerate(her_steps):
                    # Sample future indices (including current step to end)
                    future_indices = np.random.randint(t, len(her_steps), size=k_future)
                    
                    for f_idx in future_indices:
                        future_pos = her_steps[f_idx]["pos_after"]
                        pseudo_goal = future_pos.copy()

                        pb = step["pos_before"]
                        pa = step["pos_after"]
                        act = step["action"]

                        s_her = self._get_hl_state_for_goal(pb, pseudo_goal)
                        s_next_her = self._get_hl_state_for_goal(pa, pseudo_goal)

                        dist_next = np.linalg.norm(pa - pseudo_goal)
                        if dist_next < self.args.main_goal_success_radius:
                            r_her = self.args.hl_success_bonus
                            d_her = True
                        else:
                            r_her = -self.args.hl_time_penalty
                            d_her = False

                        self.agent.store(s_her, act, r_her, s_next_her, d_her)
                        
                        # Optional: Update on HER data immediately (can be computationally expensive)
                        # To save time, we can update less frequently or just rely on the main loop updates
                        # But for sample efficiency, we update here.
                        loss = self.agent.update(self.args.hl_updates_per_step)
                        if loss is not None:
                            ep_losses.append(loss)

            if final_main_dist is None:
                pos = self.get_ee_pos(self.env.arm_angles)
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
        a_pos = self.get_ee_pos(self.env.arm_angles)

        self.main_goal = self._sample_main_goal(a_pos)
        self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

        agent_traj = []
        subgoals = []
        main_dists = []

        for hl_step in range(self.args.max_high_steps_eval):
            pos = self.get_ee_pos(self.env.arm_angles)

            agent_traj.append(pos.copy())
            main_dists.append(np.linalg.norm(pos - self.main_goal))

            s_h = self._get_hl_state_for_goal(pos, self.main_goal)
            action = self.agent.select_action(s_h, greedy=True)

            subgoal = pos + action
            subgoals.append(subgoal.copy())

            self.env.goal_position = np.array(subgoal, dtype=np.float32)
            for _ in range(self.args.low_horizon_eval):
                obs_l = self.env._get_observation()
                obs_l = np.nan_to_num(obs_l, nan=0.0)
                ll_action = self._low_level_policy(obs_l, subgoal)
                _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                pos = self.get_ee_pos(self.env.arm_angles)
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
            plt.plot(agent_traj[:, 0], agent_traj[:, 2], marker="o", label="Agent EE")
        if len(subgoals) > 0:
            plt.scatter(subgoals[:, 0], subgoals[:, 2], marker="x", s=80, label="Subgoals")
        plt.scatter([main_goal[0]], [main_goal[2]], marker="*", s=200, label="Main goal")
        plt.xlabel("X (world)")
        plt.ylabel("Z (world)")
        plt.title("HL debug: Arm EE Trajectory (X-Z)")
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
            a_pos = self.get_ee_pos(self.env.arm_angles)

            self.main_goal = self._sample_main_goal(a_pos)
            self.env.goal_position = np.array(self.main_goal, dtype=np.float32)

            ep_success = False

            for hl_step in range(self.args.max_high_steps_eval):
                pos = self.get_ee_pos(self.env.arm_angles)

                s_h = self._get_hl_state_for_goal(pos, self.main_goal)
                action = self.agent.select_action(s_h, greedy=True)

                subgoal = pos + action
                self.env.goal_position = np.array(subgoal, dtype=np.float32)

                for _ in range(self.args.low_horizon_eval):
                    obs_l = self.env._get_observation()
                    obs_l = np.nan_to_num(obs_l, nan=0.0)
                    ll_action = self._low_level_policy(obs_l, subgoal)
                    _, _, done_l, trunc_l, _ = self.env.step(ll_action)

                    pos = self.get_ee_pos(self.env.arm_angles)
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
        description="Continuous HAC-style Manipulation + HER"
    )

    # general
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=500,
                   help="HL training episodes")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="hac_continuous_her_arm_models")

    # low-level PPO
    p.add_argument("--low_model_path", type=str,
                   default="logs/simple_arm/realistic_sac_20251207_062739/final_sac")
    p.add_argument("--low_total_timesteps", type=int, default=250_000)
    p.add_argument("--skip_low_train", action="store_true",
                   help="Skip low-level PPO training if model exists")
    p.add_argument("--low_tensorboard_log", type=str,
                   default="./logs_lowlevel/")
    p.add_argument("--low_checkpoint_dir", type=str,
                   default="./models/lowlevel_checkpoints/")
    p.add_argument("--low_best_dir", type=str,
                   default="./models/lowlevel_best/")
    p.add_argument("--low_model_type", type=str, default="SAC")
    p.add_argument("--low_action_scale", type=float, default=0.3,
                   help="Scale for low-level actions (0.3m per action step)")

    # main goal
    p.add_argument("--main_goal_min_dist", type=float, default=0.3,
                   help="Min distance for arm reaching (0.3m reachable)")
    p.add_argument("--main_goal_max_dist", type=float, default=2.0,
                   help="Max distance for arm reaching (2m workspace)")
    p.add_argument("--main_goal_success_radius", type=float, default=0.3,
                   help="Success threshold (arm EE precision)")

    # subgoals
    p.add_argument("--subgoal_base_step", type=float, default=0.5,
                   help="Base subgoal step (0.5m for arm)")
    p.add_argument("--subgoal_offset_scale", type=float, default=2.0)
    p.add_argument("--subgoal_success_radius", type=float, default=0.15) # used
    p.add_argument("--min_subgoal_movement", type=float, default=1.0)

    # horizons
    p.add_argument("--max_high_steps", type=int, default=20)
    p.add_argument("--low_horizon", type=int, default=50)
    p.add_argument("--max_high_steps_eval", type=int, default=20)
    p.add_argument("--low_horizon_eval", type=int, default=50)

    # HL TD3 hyperparams
    p.add_argument("--hl_actor_lr", type=float, default=1e-3)
    p.add_argument("--hl_critic_lr", type=float, default=1e-3)
    p.add_argument("--hl_gamma", type=float, default=0.99)
    p.add_argument("--hl_tau", type=float, default=0.005)
    p.add_argument("--hl_buffer", type=int, default=100_000)
    p.add_argument("--hl_batch", type=int, default=256)
    p.add_argument("--hl_init_noise_std", type=float, default=0.3)
    p.add_argument("--hl_min_noise_std", type=float, default=0.05)
    p.add_argument("--hl_noise_decay_episodes", type=int, default=5000)
    p.add_argument("--hl_updates_per_step", type=int, default=2)

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

    # # ---- Stage 1: low-level PPO ----
    # low_model_zip = args.low_model_path + ".zip"
    # if args.skip_low_train and os.path.exists(low_model_zip):
    #     print(f"Skipping low-level training, found {low_model_zip}")
    # else:
    #     train_low_level_ppo(args)

    # ---- Stage 2: high-level TD3 + HER ----
    env = HabitatArmReachingEnv(max_steps=200)

    # HL state = [agent_x, agent_y, agent_z, goal_x, goal_y, goal_z, delta_x, delta_y, delta_z] -> dim=9
    state_dim = 9
    action_dim = 3  # 3D offset in x-y-z

    hl_agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=args.hl_actor_lr,
        critic_lr=args.hl_critic_lr,
        gamma=args.hl_gamma,
        tau=args.hl_tau,
        buffer_capacity=args.hl_buffer,
        batch_size=args.hl_batch,
        device=device,
        max_action=args.subgoal_offset_scale,
        init_noise_std=args.hl_init_noise_std,
        min_noise_std=args.hl_min_noise_std,
        noise_decay_episodes=args.hl_noise_decay_episodes,
    )

    if args.low_model_type == "PPO":
        ll_agent = PPO.load(args.low_model_path, device=device)
    elif args.low_model_type == "SAC":
        ll_agent = SAC.load(args.low_model_path, device=device)
    elif args.low_model_type == "A2C":
        ll_agent = A2C.load(args.low_model_path, device=device)
    else:
        raise ValueError(f"Unsupported low_model_type: {args.low_model_type}")

    trainer = HighLevelTD3HERTrainer(
        env=env,
        ll_agent=ll_agent,
        hl_agent=hl_agent,
        args=args,
    )

    trainer.train()

    # save HL weights
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp+"_"+args.low_model_type)
    os.makedirs(save_dir, exist_ok=True)
    hl_actor_path = os.path.join(save_dir, "hl_actor.pth")
    hl_critic_1_path = os.path.join(save_dir, "hl_critic_1.pth")
    hl_critic_2_path = os.path.join(save_dir, "hl_critic_2.pth")
    torch.save(hl_agent.actor.state_dict(), hl_actor_path)
    torch.save(hl_agent.critic_1.state_dict(), hl_critic_1_path)
    torch.save(hl_agent.critic_2.state_dict(), hl_critic_2_path)
    print(f"Saved HL actor to  {hl_actor_path}")
    print(f"Saved HL critic_1 to {hl_critic_1_path}")
    print(f"Saved HL critic_2 to {hl_critic_2_path}")

    trainer.plot_training_curves(save_dir)
    debug_prefix = os.path.join(save_dir, "hl_debug")
    trainer.debug_episode_trajectory(save_prefix=debug_prefix)

    trainer.evaluate(args.eval_episodes)

    env.close()
    print("Done. Outputs saved in:", save_dir)


if __name__ == "__main__":
    main()
