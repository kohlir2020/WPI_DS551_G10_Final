"""
Example: How to use the trained navigation model programmatically.

This shows different ways to integrate the trained model into your HRL pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')

from test_nav import FetchNavigationVisualizer
import numpy as np


def example_1_simple_episode():
    """Example 1: Run a single episode with visualization."""
    print("\n" + "="*60)
    print("Example 1: Simple Episode")
    print("="*60)
    
    # Create visualizer
    vis = FetchNavigationVisualizer(enable_video=False)
    
    # Run one episode
    success, steps, distance = vis.visualize_episode(display=True)
    
    print(f"\nResult: {'Success' if success else 'Failed'}")
    print(f"Steps: {steps}, Final distance: {distance:.2f}m")
    
    vis.close()


def example_2_step_by_step():
    """Example 2: Manual step-by-step control."""
    print("\n" + "="*60)
    print("Example 2: Step-by-Step Control")
    print("="*60)
    
    vis = FetchNavigationVisualizer(enable_video=False)
    
    # Reset environment
    obs = vis.reset(min_goal_dist=3.0, max_goal_dist=5.0)
    print(f"Initial observation: distance={obs[0]:.2f}m, angle={obs[1]:.2f}rad")
    
    # Run for specific number of steps
    max_steps = 50
    for step in range(max_steps):
        # Get action from model
        action, _ = vis.model.predict(obs, deterministic=True)
        action = int(action)
        
        # Step environment
        obs, distance, done, info = vis.step(action)
        
        action_names = ["NO-OP", "FORWARD", "LEFT", "RIGHT"]
        print(f"Step {step+1:3d}: {action_names[action]:7s} -> distance={distance:.2f}m")
        
        if done:
            print(f"\n✓ Goal reached in {step+1} steps!")
            break
    else:
        print(f"\n✗ Goal not reached in {max_steps} steps")
    
    vis.close()


def example_3_custom_goal():
    """Example 3: Navigate to specific goal coordinates."""
    print("\n" + "="*60)
    print("Example 3: Custom Goal Navigation")
    print("="*60)
    
    vis = FetchNavigationVisualizer(enable_video=False)
    
    # Reset to get a valid start position
    obs = vis.reset()
    
    # Set custom goal position
    custom_goal = np.array([5.0, 0.0, -3.0], dtype=np.float32)
    
    # Make sure goal is navigable
    if vis.pathfinder.is_navigable(custom_goal):
        vis.goal_position = custom_goal
        print(f"✓ Custom goal set: {custom_goal}")
    else:
        # Snap to nearest navigable point
        vis.goal_position = vis.pathfinder.snap_point(custom_goal)
        print(f"✓ Goal snapped to navigable point: {vis.goal_position}")
    
    # Recalculate observation with new goal
    obs = vis._get_observation()
    print(f"Distance to custom goal: {obs[0]:.2f}m")
    
    # Navigate to goal
    done = False
    step = 0
    while not done and step < 200:
        action, _ = vis.model.predict(obs, deterministic=True)
        obs, distance, done, info = vis.step(int(action))
        step += 1
        
        if step % 10 == 0:
            print(f"  Step {step}: distance={distance:.2f}m")
    
    if info['success']:
        print(f"\n✓ Reached custom goal in {step} steps")
    else:
        print(f"\n✗ Did not reach custom goal ({distance:.2f}m away)")
    
    vis.close()


def example_4_extract_trajectory():
    """Example 4: Extract and analyze agent trajectory."""
    print("\n" + "="*60)
    print("Example 4: Trajectory Analysis")
    print("="*60)
    
    vis = FetchNavigationVisualizer(enable_video=False)
    
    # Reset
    obs = vis.reset()
    
    # Track trajectory
    trajectory = []
    distances = []
    actions = []
    
    done = False
    while not done:
        # Record current position
        state = vis.agent.get_state()
        trajectory.append(state.position.copy())
        distances.append(obs[0])
        
        # Get and execute action
        action, _ = vis.model.predict(obs, deterministic=True)
        actions.append(int(action))
        
        obs, distance, done, info = vis.step(int(action))
    
    # Analyze trajectory
    trajectory = np.array(trajectory)
    distances = np.array(distances)
    
    print(f"\nTrajectory Analysis:")
    print(f"  Total steps: {len(trajectory)}")
    print(f"  Start position: {trajectory[0]}")
    print(f"  End position: {trajectory[-1]}")
    print(f"  Start distance: {distances[0]:.2f}m")
    print(f"  End distance: {distances[-1]:.2f}m")
    print(f"  Distance improvement: {distances[0] - distances[-1]:.2f}m")
    
    # Action distribution
    action_names = ["NO-OP", "FORWARD", "LEFT", "RIGHT"]
    for i, name in enumerate(action_names):
        count = sum(1 for a in actions if a == i)
        pct = count / len(actions) * 100
        print(f"  {name}: {count} times ({pct:.1f}%)")
    
    vis.close()
    
    return trajectory, distances, actions


def example_5_multiple_episodes_statistics():
    """Example 5: Run multiple episodes and collect statistics."""
    print("\n" + "="*60)
    print("Example 5: Multi-Episode Statistics")
    print("="*60)
    
    vis = FetchNavigationVisualizer(enable_video=False)
    
    num_episodes = 10
    results = []
    
    for ep in range(num_episodes):
        obs = vis.reset(min_goal_dist=4.0, max_goal_dist=7.0)
        start_dist = obs[0]
        
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = vis.model.predict(obs, deterministic=True)
            obs, distance, done, info = vis.step(int(action))
            steps += 1
        
        results.append({
            'episode': ep + 1,
            'success': info['success'],
            'steps': steps,
            'start_distance': start_dist,
            'final_distance': distance
        })
        
        status = "✓" if info['success'] else "✗"
        print(f"Ep {ep+1:2d}: {status} | {steps:3d} steps | "
              f"{start_dist:.2f}m -> {distance:.2f}m")
    
    # Compute statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_episodes * 100
    
    successful = [r for r in results if r['success']]
    if successful:
        avg_steps = np.mean([r['steps'] for r in successful])
        print(f"\nSuccess Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
        print(f"Average Steps (successful): {avg_steps:.1f}")
    else:
        print(f"\nSuccess Rate: 0% - No successful episodes")
    
    vis.close()
    
    return results


def example_6_hrl_integration():
    """Example 6: Integration with high-level planner (simulated)."""
    print("\n" + "="*60)
    print("Example 6: HRL Integration Simulation")
    print("="*60)
    
    vis = FetchNavigationVisualizer(enable_video=False)
    
    # Simulate high-level plan with waypoints
    waypoints = [
        np.array([2.0, 0.0, 3.0]),
        np.array([5.0, 0.0, 1.0]),
        np.array([3.0, 0.0, -2.0]),
    ]
    
    print(f"High-level plan: Navigate through {len(waypoints)} waypoints\n")
    
    # Start from random position
    obs = vis.reset()
    start_pos = vis.agent.get_state().position
    print(f"Starting position: {start_pos}")
    
    total_steps = 0
    
    for i, waypoint in enumerate(waypoints):
        print(f"\n→ Waypoint {i+1}: {waypoint}")
        
        # Snap waypoint to navigable point
        navigable_waypoint = vis.pathfinder.snap_point(waypoint)
        vis.goal_position = navigable_waypoint
        
        # Navigate to waypoint
        obs = vis._get_observation()
        print(f"  Distance: {obs[0]:.2f}m")
        
        done = False
        steps = 0
        max_steps_per_waypoint = 100
        
        while not done and steps < max_steps_per_waypoint:
            action, _ = vis.model.predict(obs, deterministic=True)
            obs, distance, done, info = vis.step(int(action))
            steps += 1
        
        total_steps += steps
        
        if info['success']:
            print(f"  ✓ Reached waypoint in {steps} steps")
        else:
            print(f"  ✗ Could not reach waypoint ({distance:.2f}m away)")
            print(f"  Proceeding to next waypoint anyway...")
    
    final_pos = vis.agent.get_state().position
    print(f"\nFinal position: {final_pos}")
    print(f"Total steps: {total_steps}")
    
    vis.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation model usage examples")
    parser.add_argument("--example", type=int, default=0, 
                       help="Which example to run (0=all, 1-6=specific)")
    args = parser.parse_args()
    
    examples = [
        example_1_simple_episode,
        example_2_step_by_step,
        example_3_custom_goal,
        example_4_extract_trajectory,
        example_5_multiple_episodes_statistics,
        example_6_hrl_integration,
    ]
    
    if args.example == 0:
        # Run all examples
        for i, example_func in enumerate(examples, 1):
            try:
                example_func()
                input(f"\nPress Enter to continue to Example {i+1}..." if i < len(examples) else "\nPress Enter to exit...")
            except KeyboardInterrupt:
                print("\nExamples interrupted by user")
                break
    elif 1 <= args.example <= len(examples):
        # Run specific example
        examples[args.example - 1]()
    else:
        print(f"Invalid example number. Choose 0-{len(examples)}")
