# hrl/hac_agent.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import Actor, Critic
from .replay_buffer import HindsightReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HACAgent:
    def __init__(
        self,
        obs_dim,
        goal_dim,
        act_dim,
        subgoal_dim,
        subgoal_horizon=10,
        gamma=0.99,
        lr=3e-4,
        buffer_size=100000,
        batch_size=256,
    ):
        # Level 0 (low-level): goal = subgoal (2D waypoint)
        self.l0_actor = Actor(obs_dim, subgoal_dim, act_dim).to(device)
        self.l0_critic = Critic(obs_dim, subgoal_dim, act_dim).to(device)
        self.l0_actor_opt = Adam(self.l0_actor.parameters(), lr=lr)
        self.l0_critic_opt = Adam(self.l0_critic.parameters(), lr=lr)

        # Level 1 (high-level): goal = final task goal (2D position)
        # Action = subgoal in same space (2D)
        self.l1_actor = Actor(obs_dim, goal_dim, subgoal_dim).to(device)
        self.l1_critic = Critic(obs_dim, goal_dim, subgoal_dim).to(device)
        self.l1_actor_opt = Adam(self.l1_actor.parameters(), lr=lr)
        self.l1_critic_opt = Adam(self.l1_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.subgoal_horizon = subgoal_horizon

        # Replay buffers for each level
        self.l0_buf = HindsightReplayBuffer(buffer_size, obs_dim, subgoal_dim, 1)
        self.l1_buf = HindsightReplayBuffer(buffer_size, obs_dim, goal_dim, subgoal_dim)

        # Success / reward funcs used for HER
        self.success_func = lambda ag, g: np.linalg.norm(ag - g) < 0.3
        self.reward_func = lambda ag, g: 1.0 if self.success_func(ag, g) else -np.linalg.norm(ag - g)

        self.act_dim = act_dim
        self.subgoal_dim = subgoal_dim

    def _onehot(self, indices, dim):
        b = indices.shape[0]
        out = torch.zeros(b, dim, device=device)
        out[torch.arange(b), indices] = 1.0
        return out

    # ---- Action selection ----

    def select_low_level_action(self, obs_vec, subgoal_vec, eps=0.1):
        obs_t = torch.from_numpy(obs_vec[None]).float().to(device)
        g_t = torch.from_numpy(subgoal_vec[None]).float().to(device)
        logits = self.l0_actor(obs_t, g_t).detach().cpu().numpy()[0]
        if np.random.rand() < eps:
            a = np.random.randint(self.act_dim)
        else:
            a = int(np.argmax(logits))
        return a

    def select_high_level_subgoal(self, obs_vec, goal_vec, noise_scale=0.1):
        obs_t = torch.from_numpy(obs_vec[None]).float().to(device)
        g_t = torch.from_numpy(goal_vec[None]).float().to(device)
        subgoal = self.l1_actor(obs_t, g_t).detach().cpu().numpy()[0]
        subgoal = subgoal + noise_scale * np.random.randn(self.subgoal_dim)
        return subgoal.astype(np.float32)

    # ---- Experience collection (one episode) ----

    def run_episode(self, env, max_high_level_steps=5):
        """
        env: HabitatGoalNavEnv
        Assumes env.desired_goal already set or env.reset() sets one.
        """
        obs_dict = env.reset()
        obs_vec = obs_dict["observation"]
        final_goal = obs_dict["desired_goal"]

        episode_reward = 0.0
        success = False

        for hl_step in range(max_high_level_steps):
            # Sample subgoal from level 1 policy
            subgoal = self.select_high_level_subgoal(obs_vec, final_goal)

            # Rollout low-level policy for up to subgoal_horizon steps
            hl_start_obs = obs_vec.copy()
            hl_start_pos = obs_dict["achieved_goal"].copy()

            for t in range(self.subgoal_horizon):
                action = self.select_low_level_action(obs_vec, subgoal)
                next_obs_dict, _, done, info = env.step(action)

                next_obs_vec = next_obs_dict["observation"]
                ag = obs_dict["achieved_goal"]
                next_ag = next_obs_dict["achieved_goal"]

                # Level 0 transition: goal = subgoal
                r0 = self.reward_func(next_ag, subgoal)
                d0 = float(self.success_func(next_ag, subgoal))

                self.l0_buf.add(
                    obs_vec,
                    subgoal,
                    np.array([action], dtype=np.int64),
                    np.array([r0], dtype=np.float32),
                    next_obs_vec,
                    subgoal,
                    np.array([d0], dtype=np.float32),
                    ag,
                    next_ag,
                )

                obs_vec = next_obs_vec
                obs_dict = next_obs_dict
                episode_reward += r0

                # EARLY EXIT if subgoal achieved
                if d0 > 0.5:
                    break

            # Level 1 transition (coarse): from hl_start_obs to current obs_vec
            hl_next_obs_vec = obs_vec
            hl_ag = hl_start_pos
            hl_next_ag = obs_dict["achieved_goal"]
            r1 = self.reward_func(hl_next_ag, final_goal)
            d1 = float(self.success_func(hl_next_ag, final_goal))

            self.l1_buf.add(
                hl_start_obs,
                final_goal,
                subgoal,
                np.array([r1], dtype=np.float32),
                hl_next_obs_vec,
                final_goal,
                np.array([d1], dtype=np.float32),
                hl_ag,
                hl_next_ag,
            )

            if d1 > 0.5:
                success = True
                break

        return episode_reward, success

    # ---- Training steps ----

    def train_step_level(self, level=0):
        if level == 0:
            buf = self.l0_buf
            goal_dim = self.subgoal_dim
            actor = self.l0_actor
            critic = self.l0_critic
            a_opt = self.l0_actor_opt
            c_opt = self.l0_critic_opt
            her_goal_dim = self.subgoal_dim
        else:
            buf = self.l1_buf
            goal_dim = 2
            actor = self.l1_actor
            critic = self.l1_critic
            a_opt = self.l1_actor_opt
            c_opt = self.l1_critic_opt
            her_goal_dim = goal_dim

        if buf.size < self.batch_size:
            return {}

        batch = buf.sample_batch(
            self.batch_size,
            her_ratio=0.8,
            success_func=self.success_func,
            reward_func=self.reward_func,
        )

        obs = torch.from_numpy(batch["obs"]).float().to(device)
        goal = torch.from_numpy(batch["goal"]).float().to(device)
        act = torch.from_numpy(batch["act"]).long().to(device)
        rew = torch.from_numpy(batch["rew"]).float().to(device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(device)
        next_goal = torch.from_numpy(batch["next_goal"]).float().to(device)
        done = torch.from_numpy(batch["done"]).float().to(device)

        act_onehot = self._onehot(act.squeeze(-1), self.act_dim)

        # Critic target: r + γ * V(s', g')
        with torch.no_grad():
            next_logits = actor(next_obs, next_goal)
            next_actions = torch.argmax(next_logits, dim=-1)
            next_onehot = self._onehot(next_actions, self.act_dim)
            target_q = critic(next_obs, next_goal, next_onehot)
            target = rew + self.gamma * (1.0 - done) * target_q

        # Critic loss
        q = critic(obs, goal, act_onehot)
        critic_loss = F.mse_loss(q, target)

        c_opt.zero_grad()
        critic_loss.backward()
        c_opt.step()

        # Actor loss: max Q (so minimize -Q)
        logits = actor(obs, goal)
        actions = torch.argmax(logits, dim=-1)
        actions_1h = self._onehot(actions, self.act_dim)
        actor_loss = -critic(obs, goal, actions_1h).mean()

        a_opt.zero_grad()
        actor_loss.backward()
        a_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def train_both_levels(self, steps_l0=1, steps_l1=1):
        stats = {}
        for _ in range(steps_l0):
            s0 = self.train_step_level(level=0)
            stats.update({f"l0_{k}": v for k, v in s0.items()})
        for _ in range(steps_l1):
            s1 = self.train_step_level(level=1)
            stats.update({f"l1_{k}": v for k, v in s1.items()})
        return stats
