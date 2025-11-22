# test_multiple_episodes.py
from simple_navigation_env import SimpleNavigationEnv

env = SimpleNavigationEnv()

successes = 0
for ep in range(10):
    obs, _ = env.reset()
    print(f"\nEpisode {ep+1}: Start distance = {obs[0]:.2f}m")
    
    for step in range(500):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            print(f"  ✓ SUCCESS in {step} steps!")
            successes += 1
            break
        if truncated:
            print(f"  ✗ Timeout (still {info['distance']:.2f}m away)")
            break

print(f"\n{successes}/10 episodes succeeded with random actions")
env.close()