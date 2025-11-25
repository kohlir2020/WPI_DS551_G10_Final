from hrl_highlevel_env import HRLHighLevelEnv
from stable_baselines3 import PPO
import numpy as np

MODEL_PATH = "models/highlevel_manager_final"

env = HRLHighLevelEnv(
    low_level_model_path="models/highlevel_manager_final",
    subgoal_distance=3.0,
    option_horizon=100,
)

model = PPO.load(MODEL_PATH)

print("\n==============================")
print("   HRL HIGH-LEVEL EVALUATION  ")
print("==============================\n")

obs, _ = env.reset()
main_goal = env.main_goal
agent_start = env.ll_env.agent.get_state().position

print(f"Start Position : {agent_start}")
print(f"Main Goal      : {main_goal}\n")

for step in range(20):

    action, _ = model.predict(obs, deterministic=True)

    print(f"\nHL Step {step+1}:")
    print(f"  Action (direction index): {action}")

    obs, reward, done, truncated, info = env.step(action)

    print(f"  New Obs        : {obs}")
    print(f"  Reward         : {reward:.3f}")
    print(f"  Distance to Goal: {info['main_distance']:.3f}")
    print(f"HL Obs: {obs}")
    print(f"Chosen Action: {action}")
    print(f"Reward: {reward:.3f}")
    print(f"Main Dist: {info['main_distance']:.3f}")


    if done:
        print("\n>>> MAIN GOAL REACHED! <<<")
        break
    if truncated:
        print("\n>>> MAX HL STEPS REACHED <<<")
        break

env.close()
print("\nEvaluation complete.")
