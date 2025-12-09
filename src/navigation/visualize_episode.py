# visualize_episode.py
from simple_navigation_env import SimpleNavigationEnv
import matplotlib.pyplot as plt

env = SimpleNavigationEnv()
obs, _ = env.reset()

distances = [obs[0]]
actions_taken = []
action_colors = {0: 'gray', 1: 'green', 2: 'blue', 3: 'orange'}

print(f"Starting distance: {obs[0]:.2f}m\n")

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    distances.append(info['distance'])
    action_names = ["STOP", "FWD", "LEFT", "RIGHT"]
    actions_taken.append(action_names[action])
    
    if done:
        print(f"Reached goal in {step} steps!")
        break
# colors = [action_colors[a] for a in actions_taken]

# for episode in range(5):
#     distances = run_episode(model)
#     plt.plot(distances, alpha=0.5, label=f'Episode {episode}')
# plt.legend()
# env.close()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(distances, marker='o')
plt.axhline(y=0.5, color='r', linestyle='--', label='Success threshold')
plt.xlabel('Step')
plt.ylabel('Distance to Goal (m)')
plt.title('Navigation Episode')
plt.legend()
plt.grid(True)
# plt.scatter(range(len(distances)), distances, c=colors, s=50)
# plt.legend(['STOP', 'FWD', 'LEFT', 'RIGHT'])
plt.savefig('episode_visualization.png')
print("\nSaved plot to episode_visualization.png")