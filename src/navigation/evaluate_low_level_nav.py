from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv
import numpy as np
import glob, os

env = SimpleNavigationEnv()

# ============================================================
# Load model (final or best checkpoint)
# ============================================================
model_path = None

if os.path.exists("models/lowlevel_curriculum_250k.zip"):
    model_path = "models/lowlevel_curriculum_250k"
elif os.path.exists("models/lowlevel_curriculum_250k.zip"):
    model_path = "models/lowlevel_curriculum_250k.zip"
else:
    best = max(glob.glob("models/checkpoints/*.zip"), key=os.path.getctime)
    model_path = best

print(f"Loading: {model_path}")
model = PPO.load(model_path)

# ============================================================
# Evaluation settings
# ============================================================
EPISODES = 20
successes = 0
steps_list = []
distance_logs = []  # for graphs if needed

log_lines = []  # for saving readable TXT

print("\nEvaluating low-level navigation skill...\n")

for ep in range(EPISODES):
    obs, _ = env.reset()
    start_dist = obs[0]
    dist_curve = [start_dist]

    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        dist_curve.append(info["distance"])

        if done:
            successes += 1
            steps_list.append(step + 1)

            line = f"Ep {ep+1:02d}: SUCCESS in {step+1} steps | start={start_dist:.2f}m | final={info['distance']:.2f}m"
            print(line)
            log_lines.append(line)
            break

        if truncated:
            line = f"Ep {ep+1:02d}: TIMEOUT | start={start_dist:.2f}m | final={info['distance']:.2f}m"
            print(line)
            log_lines.append(line)
            break

    distance_logs.append(np.array(dist_curve, dtype=np.float32))

# ============================================================
# Results summary
# ============================================================
success_rate = successes / EPISODES * 100
avg_steps = np.mean(steps_list) if steps_list else 0

print("\n====================== RESULTS ======================")
print(f"Success Rate: {success_rate:.1f}%  ({successes}/{EPISODES})")
print(f"Average Steps to Goal: {avg_steps:.1f}")
print("=====================================================\n")

log_lines.append("\n====================== RESULTS ======================")
log_lines.append(f"Success Rate: {success_rate:.1f}%  ({successes}/{EPISODES})")
log_lines.append(f"Average Steps to Goal: {avg_steps:.1f}")
log_lines.append("=====================================================\n")

# ============================================================
# Save human-readable text log
# ============================================================
with open("evaluation_log.txt", "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print("Saved per-episode log to: evaluation_log.txt")

# ============================================================
# Save summary file
# ============================================================
with open("training_results.txt", "w") as f:
    f.write(f"Success Rate: {success_rate:.1f}%\n")
    f.write(f"Average Steps: {avg_steps:.1f}\n")
    f.write(f"Improvement over Random: {success_rate - 10:.1f}%\n")

print("Saved summary to: training_results.txt")

# ============================================================
# Optional: save raw distance logs for graphing later
# ============================================================
np.save("distance_logs.npy", np.array(distance_logs, dtype=object))
print("Saved distance curve logs to distance_logs.npy")

env.close()
