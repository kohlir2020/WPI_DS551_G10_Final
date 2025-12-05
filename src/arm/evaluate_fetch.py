# evaluate_fetch.py
"""
Evaluate trained Fetch navigation policy
"""

from stable_baselines3 import PPO
from fetch_navigation_env import FetchNavigationEnv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def evaluate_policy(model_path=None, n_episodes=20):
    """Evaluate trained policy"""
    
    print("\n" + "="*60)
    print("EVALUATING FETCH NAVIGATION POLICY")
    print("="*60 + "\n")
    
    # Load model
    if model_path is None:
        # Try to find latest model
        if os.path.exists("models/fetch/fetch_nav_100k_final.zip"):
            model_path = "models/fetch/fetch_nav_100k_final.zip"
        else:
            # Find latest checkpoint
            checkpoints = glob.glob("models/fetch/checkpoints/*.zip")
            if not checkpoints:
                print("❌ No trained model found!")
                print("Train first: python train_fetch_nav.py")
                return
            model_path = max(checkpoints, key=os.path.getctime)
    
    print(f"Loading model: {model_path}\n")
    
    env = FetchNavigationEnv()
    model = PPO.load(model_path)
    
    # Run episodes
    successes = 0
    steps_to_goal = []
    distances_over_time = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        start_dist = obs[0]
        ep_distances = [start_dist]
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_distances.append(info['distance'])
            
            if done:
                print(f"Ep {ep+1:2d}: ✓ SUCCESS in {step:3d} steps "
                      f"(started {start_dist:.2f}m away)")
                successes += 1
                steps_to_goal.append(step)
                distances_over_time.append(ep_distances)
                break
            if truncated:
                print(f"Ep {ep+1:2d}: ✗ TIMEOUT at {info['distance']:.2f}m "
                      f"(started {start_dist:.2f}m away)")
                distances_over_time.append(ep_distances)
                break
    
    env.close()
    
    # Calculate metrics
    success_rate = successes / n_episodes * 100
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else 0
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{n_episodes})")
    if steps_to_goal:
        print(f"Average Steps to Goal: {avg_steps:.1f}")
        print(f"Min Steps: {min(steps_to_goal)}")
        print(f"Max Steps: {max(steps_to_goal)}")
    print(f"{'='*60}\n")
    
    # Plot results
    plot_results(distances_over_time, success_rate, successes, n_episodes)
    
    # Save results
    with open("results/fetch_nav_results.txt", "w") as f:
        f.write(f"Fetch Navigation Results\n")
        f.write(f"="*40 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        if steps_to_goal:
            f.write(f"Avg Steps: {avg_steps:.1f}\n")
        f.write(f"\nRandom Baseline: ~10%\n")
        f.write(f"Improvement: {success_rate - 10:.1f}%\n")
    
    print("Results saved to results/fetch_nav_results.txt")
    
    return success_rate, avg_steps


def plot_results(distances_over_time, success_rate, successes, n_episodes):
    """Create visualization of results"""
    
    os.makedirs("results", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distance over time for first 10 episodes
    ax1 = axes[0]
    for i, distances in enumerate(distances_over_time[:10]):
        label = f"Ep {i+1}"
        ax1.plot(distances, alpha=0.6, label=label)
    
    ax1.axhline(y=0.5, color='r', linestyle='--', 
                label='Success threshold', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Distance to Goal (m)')
    ax1.set_title('Distance Convergence (First 10 Episodes)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rate comparison
    ax2 = axes[1]
    categories = ['Random\nBaseline', 'Trained\nFetch']
    rates = [10, success_rate]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax2.bar(categories, rates, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Navigation Performance\n({successes}/{n_episodes} successful)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/fetch_nav_evaluation.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved to results/fetch_nav_evaluation.png\n")
    plt.close()


def compare_random_baseline(n_episodes=10):
    """Test random policy as baseline"""
    
    print("\n" + "="*60)
    print("RANDOM BASELINE EVALUATION")
    print("="*60 + "\n")
    
    env = FetchNavigationEnv()
    successes = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        start_dist = obs[0]
        
        for step in range(500):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print(f"Ep {ep+1:2d}: ✓ SUCCESS (random luck!)")
                successes += 1
                break
            if truncated:
                print(f"Ep {ep+1:2d}: ✗ TIMEOUT")
                break
    
    env.close()
    
    success_rate = successes / n_episodes * 100
    print(f"\nRandom Success Rate: {success_rate:.1f}%")
    print("="*60 + "\n")
    
    return success_rate


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        # Evaluate random baseline
        compare_random_baseline()
    else:
        # Evaluate trained policy
        evaluate_policy()
