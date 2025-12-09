# src/hac/eval_skill_hac.py
import argparse
import os
import numpy as np
import torch
import habitat

from envs.habitat_goal_nav_env import HabitatGoalNavEnv
from envs.habitat_pick_env import HabitatPickEnv
from envs.habitat_place_env import HabitatPlaceEnv
from hrl.hac_agent import HACAgent

def make_habitat_env(config_path: str):
    cfg = habitat.get_config(config_path)
    return habitat.Env(cfg)

def build_skill_env(skill: str, config_path: str):
    hab_env = make_habitat_env(config_path)

    if skill == "nav":
        env = HabitatGoalNavEnv(hab_env, success_dist=0.3)
    elif skill == "pick":
        env = HabitatPickEnv(hab_env, success_thresh=0.05)
    elif skill == "place":
        env = HabitatPlaceEnv(hab_env, success_thresh=0.05)
    else:
        raise ValueError(f"Unknown skill: {skill}")
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=str, choices=["nav", "pick", "place"], required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tasks/rearrange/rearrange_pick.yaml",
    )
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = build_skill_env(args.skill, args.config)

    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.n
    subgoal_dim = goal_dim

    agent = HACAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        subgoal_dim=subgoal_dim,
    )

    ckpt_dir = os.path.join("checkpoints", args.skill)
    agent.l0_actor.load_state_dict(torch.load(os.path.join(ckpt_dir, "l0_actor_final.pt")))
    agent.l1_actor.load_state_dict(torch.load(os.path.join(ckpt_dir, "l1_actor_final.pt")))

    successes = 0
    rewards = []

    for ep in range(args.episodes):
        ep_reward, success = agent.run_episode(env)
        rewards.append(ep_reward)
        successes += int(success)
        print(f"[{args.skill.upper()} EVAL Ep {ep}] R={ep_reward:.2f} success={success}")

    print(
        f"{args.skill.upper()} success rate: {successes}/{args.episodes} "
        f"= {successes/args.episodes:.2f}"
    )
    print(f"{args.skill.upper()} avg reward: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    main()
