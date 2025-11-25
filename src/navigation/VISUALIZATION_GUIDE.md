# Fetch Robot Navigation Visualization Guide

## Overview

This guide explains how to visualize your trained HRL navigation agent controlling a Fetch robot in the Skokloster Castle scene using Habitat-Sim.

## Quick Start

```bash
cd /Users/raunit/Desktop/WPI_class/WPI_DS551_G10_Final/src
python test_nav.py
```

This will:
1. Automatically find your trained model
2. Load the Skokloster Castle scene
3. Run 3 episodes with real-time visualization
4. Display RGB camera view from the Fetch robot's perspective

## Command Line Options

### Basic Usage
```bash
# Run with defaults (3 episodes, display on)
python test_nav.py

# Run more episodes
python test_nav.py --episodes 10

# Specify a specific model
python test_nav.py --model navigation/models/lowlevel_curriculum_250k.zip

# Run without display (headless mode)
python test_nav.py --no-display

# Save videos of each episode
python test_nav.py --save-video --episodes 5
```

### Advanced Usage
```bash
# Custom scene (if you have other GLB files)
python test_nav.py --scene path/to/scene.glb

# Full evaluation with video recording
python test_nav.py --episodes 20 --save-video --no-display
```

## How It Works

### 1. Model Loading

The script automatically searches for your trained model in these locations:
- `navigation/models/lowlevel_curriculum_250k.zip`
- `src/navigation/models/lowlevel_curriculum_250k.zip`
- Latest checkpoint in `models/checkpoints/`

You can also specify the exact path with `--model`.

### 2. Fetch Robot Configuration

The Fetch robot is configured with:
- **Height**: 1.5m (approximate Fetch robot height)
- **Radius**: 0.2m (collision radius)
- **RGB Camera**: 640x480 resolution, positioned at 1.25m height
- **Actions**: 
  - Action 0: NO-OP (do nothing)
  - Action 1: MOVE_FORWARD (0.5m forward)
  - Action 2: TURN_LEFT (10 degrees)
  - Action 3: TURN_RIGHT (10 degrees)

### 3. Getting Actions from Your Trained Model

Your trained PPO model takes observations and produces actions:

```python
# Observation format: [distance_to_goal, relative_angle_to_goal]
obs = np.array([5.2, 0.3])  # 5.2m away, 0.3 radians off

# Get action from trained model
action, _ = model.predict(obs, deterministic=True)
# action will be 0, 1, 2, or 3

# Step the environment
if action == 1:
    agent.act("move_forward")
elif action == 2:
    agent.act("turn_left")
elif action == 3:
    agent.act("turn_right")
```

### 4. Integration with Your HRL System

To integrate this with your high-level planner, you can use the `FetchNavigationVisualizer` class:

```python
from test_nav import FetchNavigationVisualizer

# Create visualizer
visualizer = FetchNavigationVisualizer(
    model_path="navigation/models/lowlevel_curriculum_250k.zip",
    enable_video=False
)

# Reset environment with specific goal
obs = visualizer.reset(min_goal_dist=3.0, max_goal_dist=6.0)

# Run episode
done = False
while not done:
    # Get action from your model
    action, _ = visualizer.model.predict(obs, deterministic=True)
    
    # Step environment
    obs, distance, done, info = visualizer.step(action)
    
    # Get RGB observation for visualization
    rgb_frame = visualizer._get_rgb_observation()
    
    print(f"Step {info['step']}: Distance = {distance:.2f}m")

visualizer.close()
```

## Key Classes and Methods

### FetchNavigationVisualizer

Main class that wraps Habitat-Sim and your trained model.

#### Constructor
```python
FetchNavigationVisualizer(
    model_path=None,           # Path to trained model (auto-detect if None)
    scene_path="...",          # Path to scene GLB file
    enable_video=False         # Whether to record video frames
)
```

#### Methods

**`reset(min_goal_dist=2.0, max_goal_dist=8.0)`**
- Resets the environment with a new episode
- Samples random start and goal positions
- Returns: observation `[distance, angle]`

**`step(action)`**
- Executes an action in the environment
- Args: action (0=NO-OP, 1=FORWARD, 2=LEFT, 3=RIGHT)
- Returns: `obs, distance, done, info`

**`visualize_episode(display=True)`**
- Runs one complete episode with visualization
- Returns: `success, num_steps, final_distance`

**`run_evaluation(num_episodes=5, display=True, save_videos=False)`**
- Runs multiple episodes and reports statistics
- Returns: list of results

**`save_video(filename, fps=10)`**
- Saves recorded frames as MP4 video

## Understanding the Observation Space

Your trained model expects observations in this format:

```python
obs = np.array([distance, angle], dtype=np.float32)
```

Where:
- **distance** (float): Euclidean distance to goal in meters (0 to ~50m)
- **angle** (float): Relative angle to goal in radians (-π to π)
  - Positive = goal is to the right
  - Negative = goal is to the left
  - 0 = goal is straight ahead

### How Angle is Calculated

```python
# 1. Get agent's forward direction in world frame
forward = rotate_by_quaternion(agent.rotation, [0, 0, -1])
forward[1] = 0  # Project to ground plane

# 2. Get direction to goal
to_goal = goal_position - agent_position
to_goal[1] = 0  # Project to ground plane

# 3. Compute signed angle
angle = atan2(cross_product(forward, to_goal), dot_product(forward, to_goal))
```

## Troubleshooting

### Model Not Found
```
FileNotFoundError: No trained model found
```
**Solution**: Train your model first or specify the path with `--model`

### Scene Not Found
```
FileNotFoundError: Scene not found: ...
```
**Solution**: Make sure the scene path is correct. Default expects:
`habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`

### OpenCV Display Issues
If the visualization window doesn't appear:
```bash
# Try headless mode and save videos instead
python test_nav.py --no-display --save-video
```

### GPU Memory Issues
```python
# Change to CPU rendering in the script:
backend_cfg.gpu_device_id = -1  # CPU only
```

## Example Output

```
Loading model from: navigation/models/lowlevel_curriculum_250k.zip
✓ FetchNavigationVisualizer initialized

============================================================
Fetch Robot Navigation Evaluation
Scene: Skokloster Castle
Episodes: 3
============================================================

--- Episode 1/3 ---

▶ Episode Start:
  Start position: [-3.2  0.0  5.1]
  Goal position:  [ 2.8  0.0 -1.3]
  Initial distance: 7.45m

  Step  10: FORWARD | Distance: 6.32m
  Step  20: LEFT    | Distance: 5.18m
  Step  30: FORWARD | Distance: 3.94m
  Step  40: RIGHT   | Distance: 2.61m
  Step  50: FORWARD | Distance: 1.22m
  Step  60: FORWARD | Distance: 0.43m

✓ SUCCESS
  Steps: 62
  Final distance: 0.43m

[... more episodes ...]

============================================================
SUMMARY
============================================================
Success Rate: 100.0% (3/3)
Average Steps (successful): 58.3
============================================================
```

## Tips for Better Visualization

1. **Adjust camera position**: Modify the RGB sensor position in `_setup_simulator()`:
   ```python
   rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # Higher camera
   ```

2. **Change camera angle**: Look downward:
   ```python
   rgb_sensor_spec.orientation = [0.3, 0.0, 0.0]  # Pitch down
   ```

3. **Add depth sensor**: For richer visualization:
   ```python
   depth_sensor_spec = habitat_sim.CameraSensorSpec()
   depth_sensor_spec.uuid = "depth_sensor"
   depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
   # ... configure and add to agent_cfg.sensor_specifications
   ```

4. **Adjust movement speed**: Change action amounts:
   ```python
   "move_forward": habitat_sim.agent.ActionSpec(
       "move_forward", 
       habitat_sim.agent.ActuationSpec(amount=0.25)  # Slower
   )
   ```

## Extending for Full HRL Pipeline

To use this with your high-level planner:

```python
class SkillTask:
    def __init__(self, model_path):
        self.visualizer = FetchNavigationVisualizer(
            model_path=model_path,
            enable_video=True
        )
    
    def execute_navigation_skill(self, start_pos, goal_pos):
        """Execute low-level navigation from start to goal."""
        # Set up environment
        obs = self.visualizer.reset()
        
        # Override goal
        self.visualizer.goal_position = goal_pos
        
        # Run until goal reached
        done = False
        while not done:
            action, _ = self.visualizer.model.predict(obs, deterministic=True)
            obs, distance, done, info = self.visualizer.step(action)
        
        return info['success']
```

## Related Files

- `simple_navigation_env.py` - Training environment (no visualization)
- `evaluate_low_level_nav.py` - Evaluation without visualization
- `train_low_level_nav.py` - Training script
- `hrl_highlevel_env.py` - High-level HRL environment

## Questions?

If you encounter issues or need to extend the visualization, check:
1. Habitat-Sim documentation: https://aihabitat.org/docs/habitat-sim/
2. Your training environment configuration in `simple_navigation_env.py`
3. Make sure observation space matches between training and visualization
