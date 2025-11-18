# src/hac/train_skill_hac.py
import argparse
import os
import numpy as np
import torch

import habitat

from .envs.habitat_goal_nav_env import HabitatGoalNavEnv
from .envs.habitat_pick_env import HabitatPickEnv
from .envs.habitat_place_env import HabitatPlaceEnv
from .hrl.hac_agent import HACAgent

def make_habitat_env(config_path: str):
    cfg = habitat.get_config(config_path)
    return habitat.Env(cfg)

def build_skill_env(skill: str, config_path: str):
    hab_env = make_habitat_env(config_path)

    if skill == "nav":
        env = HabitatGoalNavEnv(hab_env, success_dist=0.3)

        def success_func(ag, g):
            return np.linalg.norm(ag - g) < 0.3

        def reward_func(ag, g):
            d = np.linalg.norm(ag - g)
            return 1.0 if d < 0.3 else -d

    elif skill == "pick":
        env = HabitatPickEnv(hab_env, success_thresh=0.05)

        def success_func(ag, g):
            # For pick, success threshold in relative offset
            return np.linalg.norm(ag - g) < 0.05

        def reward_func(ag, g):
            d = np.linalg.norm(ag - g)
            return 2.0 if d < 0.05 else -d

    elif skill == "place":
        env = HabitatPlaceEnv(hab_env, success_thresh=0.05)

        def success_func(ag, g):
            return np.linalg.norm(ag - g) < 0.05

        def reward_func(ag, g):
            d = np.linalg.norm(ag - g)
            return 2.0 if d < 0.05 else -d

    else:
        raise ValueError(f"Unknown skill: {skill}")

    return env, success_func, reward_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=str, choices=["nav", "pick", "place"], required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tasks/rearrange/rearrange_pick.yaml",
        help="Habitat config path (relative to habitat-lab/habitat-lab/)",
    )
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    # Build env + task-specific reward/success functions
    env, success_func, reward_func = build_skill_env(args.skill, args.config)

    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.n
    subgoal_dim = goal_dim  # subgoal lives in same space as goal

    agent = HACAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        subgoal_dim=subgoal_dim,
        subgoal_horizon=10,
        buffer_size=200000,
        batch_size=256,
        lr=3e-4,
        success_func=success_func,
        reward_func=reward_func,
    )

    ckpt_dir = os.path.join("checkpoints", args.skill)
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(args.episodes):
        ep_reward, success = agent.run_episode(env)
        stats = agent.train_both_levels()

        print(
            f"[{args.skill.upper()} Ep {ep}] "
            f"R={ep_reward:.2f} success={success} "
            f"l0_actor={stats.get('l0_actor_loss', np.nan):.3f} "
            f"l1_actor={stats.get('l1_actor_loss', np.nan):.3f}"
        )

        if (ep + 1) % 50 == 0:
            torch.save(
                agent.l0_actor.state_dict(),
                os.path.join(ckpt_dir, f"l0_actor_ep{ep+1}.pt"),
            )
            torch.save(
                agent.l1_actor.state_dict(),
                os.path.join(ckpt_dir, f"l1_actor_ep{ep+1}.pt"),
            )

    # final save
    torch.save(agent.l0_actor.state_dict(), os.path.join(ckpt_dir, "l0_actor_final.pt"))
    torch.save(agent.l1_actor.state_dict(), os.path.join(ckpt_dir, "l1_actor_final.pt"))

if __name__ == "__main__":
    main()
