#!/usr/bin/env python3
"""
Quick evaluation of trained models on realistic environment
"""

import sys
sys.path.insert(0, 'src/arm')

from habitat_arm_reaching_env import HabitatArmReachingEnv
from stable_baselines3 import A2C, SAC
import numpy as np

def evaluate_model(model, env, num_episodes=5, model_name="Model"):
    """Evaluate trained model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"  Episode {episode+1}: Reward={episode_reward:8.2f}, Length={steps}")
    
    print(f"\n  Average Reward: {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):8.2f}")
    
    return np.mean(rewards), np.mean(episode_lengths)

def main():
    print("\n" + "="*60)
    print("MODEL EVALUATION - Realistic Arm Reaching Environment")
    print("="*60)
    
    # Create environment
    env = HabitatArmReachingEnv(
        max_steps=200,
        scene="apartment_1",
        use_habitat=False  # Use realistic fallback
    )
    
    # Load models
    a2c_model = A2C.load('logs/simple_arm/realistic_a2c_20251207_054618/final_a2c.zip')
    sac_model = SAC.load('logs/simple_arm/realistic_sac_20251207_062739/final_sac.zip')
    
    # Evaluate
    a2c_reward, a2c_length = evaluate_model(a2c_model, env, num_episodes=5, model_name="A2C (1M steps)")
    sac_reward, sac_length = evaluate_model(sac_model, env, num_episodes=5, model_name="SAC (1M steps)")
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Avg Reward':<15} {'Avg Length':<15}")
    print(f"{'-'*50}")
    print(f"{'A2C (1M)':<20} {a2c_reward:<15.2f} {a2c_length:<15.2f}")
    print(f"{'SAC (1M)':<20} {sac_reward:<15.2f} {sac_length:<15.2f}")
    print(f"{'='*60}")
    
    # Determine winner
    if a2c_reward > sac_reward:
        print(f"\n🏆 A2C performs better: +{a2c_reward - sac_reward:.2f} reward")
    elif sac_reward > a2c_reward:
        print(f"\n🏆 SAC performs better: +{sac_reward - a2c_reward:.2f} reward")
    else:
        print(f"\n🤝 Both algorithms perform similarly")
    
    env.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
