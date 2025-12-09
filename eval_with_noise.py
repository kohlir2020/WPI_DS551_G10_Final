#!/usr/bin/env python3
"""
Evaluate trained HRL model with stochastic (noisy) policy to match training.
"""
import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, "./src/arm")
os.chdir("./src/arm")

from hac_continuous_her_arm import HRL
import argparse
import numpy as np

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_dir", type=str, default="hac_continuous_her_arm_models/20251208_182832_SAC",
                      help="Path to trained model directory")
    args.add_argument("--episodes", type=int, default=20, help="Number of eval episodes")
    args_parsed = args.parse_args()
    
    # Create HRL instance with proper args
    hrl_args = argparse.Namespace(
        episodes=600,
        her_k_future=8,
        hl_progress_scale=30.0,
        hl_time_penalty=0.01,
        hl_success_bonus=150.0,
        log_interval=30,
        max_high_steps_eval=200,
        low_horizon_eval=10,
        main_goal_success_radius=0.45,
        subgoal_success_radius=0.3,
        low_action_scale=0.5,
        hl_noise_decay_episodes=500,
        hl_updates_per_step=2,
        replay_buffer_size=1000000,
        hl_batch_size=256,
        hl_actor_lr=2e-3,
        hl_critic_lr=2e-3,
        hl_tau=5e-3,
        device="cuda"
    )
    
    print(f"\n=== EVAL WITH STOCHASTIC POLICY (NOISY) ===")
    print(f"Model: {args_parsed.model_dir}")
    print(f"Episodes: {args_parsed.episodes}\n")
    
    hrl = HRL(hrl_args)
    hrl.agent.load(args_parsed.model_dir)
    hrl.evaluate(episodes=args_parsed.episodes)

if __name__ == "__main__":
    main()
