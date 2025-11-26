# HRL Navigation Visualization with Fetch Robot

## Overview

This visualization system demonstrates **Hierarchical Reinforcement Learning (HRL)** for robot navigation in the Skokloster Castle scene. The system uses a two-level hierarchy:

1. **High-Level Agent**: Plans subgoals (waypoints) using 8-directional actions
2. **Low-Level Agent**: Executes motor commands (move forward, turn left/right) to reach subgoals

## Quick Start

```bash
cd /Users/raunit/Desktop/WPI_class/WPI_DS551_G10_Final/src
conda activate habitat-lab-env
python test_nav.py
```

This will automatically load both trained models and run HRL navigation with visualization.

## HRL Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Goal                               │
│                  (Final destination)                         │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│              HIGH-LEVEL AGENT (Subgoal Planner)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input: [distance_to_main_goal, angle_to_main_goal] │   │
│  │ Output: Direction (0-7) for next subgoal           │   │
│  │ Actions: E, NE, N, NW, W, SW, S, SE                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
                    [Subgoal Generated]
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             LOW-LEVEL AGENT (Motor Controller)              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input: [distance_to_subgoal, angle_to_subgoal]     │   │
│  │ Output: Action (0-3)                                │   │
│  │ Actions: NO-OP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT│   │
│  │ Executes up to 50 steps to reach subgoal           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
                    [Fetch Robot Moves]
```

## Command Line Options

### Basic Usage

```bash
# Run with defaults (3 episodes, display on)
python test_nav.py

# Run more episodes
python test_nav.py --episodes 10

# Debug mode (detailed HRL execution info)
python test_nav.py --debug

# Save videos
python test_nav.py --save-video --episodes 5

# Headless mode
python test_nav.py --no-display
```

### Specify Models

```bash
# Use specific models
python test_nav.py \
  --low-level-model navigation/models/lowlevel_curriculum_250k.zip \
  --high-level-model navigation/models/highlevel_manager_final.zip

# Different scene
python test_nav.py --scene path/to/scene.glb
```

## How It Works

### 1. Episode Initialization

```python
# HRL environment resets both levels
obs, info = hrl_env.reset()
# obs = [distance_to_main_goal, angle_to_main_goal]

# Main goal is sampled 8-20 meters away
# Agent starts at random navigable position
```

### 2. High-Level Decision Making

```python
# High-level agent selects direction for subgoal
hl_action, _ = high_level_model.predict(obs, deterministic=True)
# hl_action ∈ {0, 1, 2, 3, 4, 5, 6, 7}

# Action mapping:
# 0: East (E)       4: West (W)
# 1: NorthEast (NE) 5: SouthWest (SW)
# 2: North (N)      6: South (S)
# 3: NorthWest (NW) 7: SouthEast (SE)
```

### 3. Subgoal Generation

The system generates a navigable subgoal 5 meters away in the chosen direction:

```python
# 1. Calculate direction based on HL action
angle = (action / 8) * 2π + bias_toward_main_goal

# 2. Compute target position
subgoal = agent_position + 5.0 * [cos(angle), 0, sin(angle)]

# 3. Snap to navigable mesh
subgoal = pathfinder.snap_point(subgoal)

# 4. Validate with pathfinding
if pathfinder.find_path(agent_pos, subgoal):
    return subgoal
else:
    # Try fallback strategies
```

### 4. Low-Level Execution

```python
# Low-level agent executes motor commands to reach subgoal
for step in range(50):  # option_horizon
    ll_obs = [distance_to_subgoal, angle_to_subgoal]
    ll_action, _ = low_level_model.predict(ll_obs, deterministic=True)
    
    if ll_action == 1:
        agent.act("move_forward")  # 0.5m
    elif ll_action == 2:
        agent.act("turn_left")     # 10°
    elif ll_action == 3:
        agent.act("turn_right")    # 10°
    
    if reached_subgoal:
        break
```

### 5. Reward Calculation

```python
progress = prev_distance_to_main_goal - current_distance_to_main_goal

reward = 10.0 * progress  # Strong progress incentive

# Bonuses
if progress > 2.0:
    reward += 5.0
elif progress > 1.0:
    reward += 2.0

# Penalties
if movement < 0.5:
    reward -= 1.0  # No movement penalty

# Success bonus
if distance_to_main_goal < 0.6:
    reward += 50.0
    done = True
```

## Example Output

```
Loading low-level model from: navigation/models/lowlevel_curriculum_250k.zip
Loading high-level model from: navigation/models/highlevel_manager_final.zip
✓ Loaded low-level skill: navigation/models/lowlevel_curriculum_250k
✓ FetchNavigationVisualizer with HRL initialized

============================================================
Fetch Robot HRL Navigation Evaluation
Scene: Skokloster Castle
Episodes: 3
High-level: Subgoal planner (8 directions)
Low-level: Motor controller (discrete actions)
============================================================

--- Episode 1/3 ---

▶ Episode Start (HRL):
  Start position: [-4.2  0.0  6.3]
  Main goal: [ 8.1  0.0 -2.7]
  Initial distance: 14.52m
  High-level action space: 8 directions

  HL Step  1: E   | Dist: 12.34m | Movement: 3.12m | Progress: +2.18m | Reward: +23.80
  HL Step  2: NE  | Dist: 10.21m | Movement: 2.89m | Progress: +2.13m | Reward: +26.30
  HL Step  3: E   | Dist: 7.45m  | Movement: 3.45m | Progress: +2.76m | Reward: +32.60
  HL Step  4: SE  | Dist: 4.67m  | Movement: 2.98m | Progress: +2.78m | Reward: +32.80
  HL Step  5: S   | Dist: 2.12m  | Movement: 2.67m | Progress: +2.55m | Reward: +30.50
  HL Step  6: SE  | Dist: 0.48m  | Movement: 1.78m | Progress: +1.64m | Reward: +66.40

✓ SUCCESS
  High-level steps: 6
  Final distance: 0.48m
  Total reward: 212.40

[... more episodes ...]

============================================================
HRL EVALUATION SUMMARY
============================================================
Success Rate: 100.0% (3/3)
Average High-Level Steps (successful): 6.7
  (Each HL step = up to 50 low-level actions)
============================================================
```

## Key Differences from Low-Level Only

| Aspect | Low-Level Only | HRL (This Implementation) |
|--------|---------------|---------------------------|
| **Decision Frequency** | Every step (~300 steps) | Every ~50 steps (~6-20 HL steps) |
| **Action Space** | 4 actions (motor commands) | 8 directions (strategic) |
| **Planning Horizon** | Immediate (next 0.5m) | Medium-term (next 5m) |
| **Observation** | Distance + angle to goal | Same, but at different scales |
| **Typical Episode Length** | 50-150 low-level steps | 5-15 high-level steps |
| **Goal Distance** | 2-8 meters | 8-20 meters |
| **Complexity** | Direct navigation | Hierarchical waypoint planning |

## Understanding the Visualization

### Display Elements

- **HL Step**: Current high-level step number (not low-level steps)
- **Action**: Direction chosen by high-level planner (E, NE, N, etc.)
- **Distance**: Remaining distance to main goal
- **Move**: How far robot moved in this high-level step
- **R**: Reward received for this step
- **GOAL**: Indicator (green when reached, orange otherwise)
- **HRL**: Badge showing this is hierarchical RL

### What You See

Each frame shows the RGB camera view from the Fetch robot's perspective (1.25m height). The robot moves through multiple low-level actions per frame, controlled by the high-level planner's subgoal decisions.

## Programmatic Usage

### Example: Run HRL Episode

```python
from test_nav import FetchNavigationVisualizer

# Create visualizer
vis = FetchNavigationVisualizer(
    low_level_model_path="navigation/models/lowlevel_curriculum_250k.zip",
    high_level_model_path="navigation/models/highlevel_manager_final.zip",
    enable_video=True,
    debug=True  # See detailed HRL execution
)

# Run episode
success, hl_steps, final_dist = vis.visualize_episode(display=True)

print(f"Success: {success}")
print(f"High-level steps: {hl_steps}")
print(f"Approximate low-level steps: {hl_steps * 50}")

vis.close()
```

### Example: Access HRL Components

```python
# Access the HRL environment
hrl_env = vis.hrl_env

# Get references to both models
low_level_model = hrl_env.low_level  # PPO model for motor control
high_level_model = vis.high_level_model  # PPO model for subgoal planning

# Access simulator
sim = vis.sim
agent = vis.agent
pathfinder = vis.pathfinder

# Get current main goal
main_goal = vis.main_goal

# Get high-level observation
hl_obs = vis._get_observation()  # [distance_to_main_goal, angle]

# Execute one high-level action
hl_action = 0  # East
obs, distance, done, info = vis.step(hl_action)
```

## Debugging HRL Execution

Run with `--debug` flag to see detailed execution:

```bash
python test_nav.py --debug
```

Output includes:
- Subgoal generation attempts and fallbacks
- Low-level action sequences
- Distance changes at each level
- Reward calculations
- Subgoal success statistics

## Model Files

The system requires two trained models:

1. **Low-Level Model** (auto-detected):
   - `navigation/models/lowlevel_curriculum_250k.zip`
   - Trained with curriculum learning for 250k steps
   - Action space: {0: NO-OP, 1: FORWARD, 2: LEFT, 3: RIGHT}
   - Observation: [distance, angle] to subgoal

2. **High-Level Model** (auto-detected):
   - `navigation/models/highlevel_manager_final.zip`
   - Trained on top of frozen low-level policy
   - Action space: {0-7} representing 8 compass directions
   - Observation: [distance, angle] to main goal

## Tips for Best Results

1. **Let it run**: HRL makes fewer but more strategic decisions
2. **Watch the movement patterns**: Robot navigates via waypoints, not direct paths
3. **Debug mode**: Use `--debug` to understand the two-level decision making
4. **Video recording**: Slower speed with `--save-video` helps see HRL planning
5. **Multiple episodes**: Success rate improves understanding of strategy diversity

## Extending the System

### Add Third Level

```python
# Meta-planner that selects among different HRL policies
class MetaPlanner:
    def __init__(self):
        self.hrl_systems = [
            FetchNavigationVisualizer(...),  # Aggressive
            FetchNavigationVisualizer(...),  # Conservative
        ]
    
    def select_policy(self, context):
        # Choose HRL system based on scene complexity
        return self.hrl_systems[0]
```

### Integrate with Task Planning

```python
# Use HRL navigation as a skill in larger task
def execute_fetch_object_task(object_location):
    # 1. Navigate to object (HRL)
    vis = FetchNavigationVisualizer(...)
    vis.hrl_env.main_goal = object_location
    success = vis.visualize_episode()
    
    # 2. Manipulate object (another HRL system)
    # ...
```

## Troubleshooting

### Models not found
```bash
# Check model directory
ls -la src/navigation/models/

# Train if missing
cd src/navigation
python train_low_level_nav.py
python train_high_level_improved.py
```

### Import errors
```bash
# Make sure you're in the right directory
cd /Users/raunit/Desktop/WPI_class/WPI_DS551_G10_Final/src

# Check Python path includes navigation
python -c "import sys; print(sys.path)"
```

### Subgoal generation failures
Enable debug mode to see which fallback strategies are being used:
```bash
python test_nav.py --debug
```
---
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
