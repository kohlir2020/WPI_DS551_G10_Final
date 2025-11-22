# evaluate_trained.py
from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv
import matplotlib.pyplot as plt

env = SimpleNavigationEnv()

# Load trained model (or latest checkpoint)
try:
    model = PPO.load("models/nav_ppo_100k_final")
    print("Loaded final model")
except:
    import glob
    latest = max(glob.glob("models/checkpoints/*.zip"), key=lambda x: os.path.getctime(x))
    model = PPO.load(latest)
    print(f"Loaded checkpoint: {latest}")

# Test 20 episodes
successes = 0
steps_to_goal = []

print("\nEvaluating trained policy on 20 episodes...\n")

for ep in range(20):
    obs, _ = env.reset()
    start_dist = obs[0]
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            print(f"Ep {ep+1:2d}: ✓ SUCCESS in {step:3d} steps (started {start_dist:.2f}m away)")
            successes += 1
            steps_to_goal.append(step)
            break
        if truncated:
            print(f"Ep {ep+1:2d}: ✗ TIMEOUT (still {info['distance']:.2f}m away)")
            break

success_rate = successes / 20 * 100
avg_steps = sum(steps_to_goal) / len(steps_to_goal) if steps_to_goal else 0

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"Success Rate: {success_rate:.1f}% ({successes}/20)")
print(f"Avg Steps to Goal: {avg_steps:.1f}")
print(f"Improvement over Random: {success_rate - 10:.1f}%")
print(f"{'='*60}")

# Save results
with open("training_results.txt", "w") as f:
    f.write(f"Success Rate: {success_rate:.1f}%\n")
    f.write(f"Average Steps: {avg_steps:.1f}\n")
    f.write(f"Improvement: {success_rate - 10:.1f}%\n")

print("\nResults saved to training_results.txt")

env.close()