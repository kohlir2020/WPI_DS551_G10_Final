# quick_train_test.py
from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv

env = SimpleNavigationEnv()

print("Quick training test (5000 steps, ~3 minutes)...\n")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    device="cpu",
)

model.learn(total_timesteps=5000, progress_bar=True)
model.save("models/test_5k")

print("\n✓ Test complete! Now test the policy:")

# Test it
obs, _ = env.reset()
print(f"\nStarting distance: {obs[0]:.2f}m")

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print(f"✓ Reached goal in {i} steps!")
        break
    if i == 99:
        print(f"✗ Didn't reach goal (distance: {info['distance']:.2f}m)")

env.close()
print("\nIf this worked, full training will work too!")