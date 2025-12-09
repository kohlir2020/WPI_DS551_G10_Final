#!/usr/bin/env python3
"""Evaluate trained model with stochastic/noisy policy."""
from hac_continuous_her_arm import HighLevelTD3HERTrainer
import argparse

# Setup args
args = argparse.Namespace(
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
    device='cuda'
)

# Load and evaluate
trainer = HighLevelTD3HERTrainer(args)
trainer.agent.load('../../hac_continuous_her_arm_models/20251208_182832_SAC')
print('\n=== EVALUATION WITH STOCHASTIC POLICY (GREEDY=FALSE) ===\n')
trainer.evaluate(episodes=20)
