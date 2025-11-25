# Quick Start: Visualizing Fetch Robot with Trained HRL Navigation

## TL;DR - Run Now

```bash
cd /Users/raunit/Desktop/WPI_class/WPI_DS551_G10_Final/src
conda activate habitat-lab-env
python test_nav.py
```

This will visualize your trained navigation agent controlling a Fetch robot in Skokloster Castle with RGB camera view.

---

## What You Need to Know

### 1. Your Trained Model

Your HRL navigation model outputs **discrete actions**:
- **0**: NO-OP (do nothing)
- **1**: MOVE_FORWARD (0.5 meters)
- **2**: TURN_LEFT (10 degrees)
- **3**: TURN_RIGHT (10 degrees)

### 2. Observation Format

Your model expects: `[distance_to_goal, angle_to_goal]`
- `distance`: Float, 0 to ~50 meters
- `angle`: Float, -π to π radians (negative=left, positive=right)

### 3. How to Get Actions

```python
from stable_baselines3 import PPO

# Load your trained model
model = PPO.load("navigation/models/lowlevel_curriculum_250k.zip")

# Get observation from environment
obs = np.array([5.2, 0.3])  # 5.2m away, 0.3 rad to right

# Get action
action, _ = model.predict(obs, deterministic=True)
# Returns: 0, 1, 2, or 3
```

### 4. How to Step the Fetch Robot

```python
import habitat_sim

# Setup (done once)
sim = habitat_sim.Simulator(config)
agent = sim.get_agent(0)

# Define actions
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", 
        habitat_sim.agent.ActuationSpec(amount=0.5)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", 
        habitat_sim.agent.ActuationSpec(amount=10.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", 
        habitat_sim.agent.ActuationSpec(amount=10.0)
    ),
}

# Execute action (each step)
if action == 1:
    agent.act("move_forward")
elif action == 2:
    agent.act("turn_left")
elif action == 3:
    agent.act("turn_right")
```

---

## Complete Example: One Episode

```python
from test_nav import FetchNavigationVisualizer

# Initialize
vis = FetchNavigationVisualizer(
    model_path="navigation/models/lowlevel_curriculum_250k.zip",
    scene_path="habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)

# Reset environment
obs = vis.reset()  # Returns [distance, angle]

# Run episode
done = False
while not done:
    # Get action from trained model
    action, _ = vis.model.predict(obs, deterministic=True)
    
    # Step environment (executes Fetch robot action)
    obs, distance, done, info = vis.step(int(action))
    
    # Get RGB camera view
    rgb_frame = vis._get_rgb_observation()
    
    print(f"Step {info['step']}: Distance = {distance:.2f}m")

# Cleanup
vis.close()
```

---

## Command Line Options

```bash
# Basic - run 3 episodes with display
python test_nav.py

# Run 10 episodes
python test_nav.py --episodes 10

# Save videos (without display)
python test_nav.py --save-video --no-display --episodes 5

# Use specific model
python test_nav.py --model path/to/model.zip

# Different scene
python test_nav.py --scene path/to/scene.glb
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/test_nav.py` | Main visualization script with Fetch robot |
| `src/navigation/simple_navigation_env.py` | Training environment (no visual rendering) |
| `src/navigation/evaluate_low_level_nav.py` | Evaluate without visualization |
| `src/navigation/example_usage.py` | Code examples for integration |
| `src/navigation/VISUALIZATION_GUIDE.md` | Detailed documentation |

---

## Fetch Robot Configuration

The Fetch robot in the simulation has:

```python
# Physical parameters
height = 1.5  # meters
radius = 0.2  # meters (collision)

# Camera sensor
resolution = [480, 640]  # height x width
position = [0.0, 1.25, 0.0]  # relative to agent base
sensor_type = RGB

# Movement
forward_speed = 0.5  # meters per step
turn_speed = 10.0    # degrees per step
```

---

## Skokloster Castle Scene

- **Location**: `habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`
- **Type**: Indoor castle environment
- **Navigable area**: Multiple rooms and corridors
- **Coordinates**: 3D positions in meters

---

## Integration with Your HRL Pipeline

### Option 1: Use as Navigation Skill

```python
class NavigationSkill:
    def __init__(self):
        self.vis = FetchNavigationVisualizer(
            model_path="navigation/models/lowlevel_curriculum_250k.zip"
        )
    
    def navigate_to_goal(self, goal_position):
        """Navigate Fetch robot to goal position."""
        obs = self.vis.reset()
        self.vis.goal_position = goal_position
        
        done = False
        while not done:
            action, _ = self.vis.model.predict(obs, deterministic=True)
            obs, distance, done, info = self.vis.step(int(action))
        
        return info['success']
```

### Option 2: Get Actions for Existing Environment

```python
# Load model
model = PPO.load("navigation/models/lowlevel_curriculum_250k.zip")

# In your environment step loop
obs = calculate_navigation_observation(agent_pos, goal_pos)
action, _ = model.predict(obs, deterministic=True)

# Execute action in your environment
if action == 1:
    agent.move_forward()
elif action == 2:
    agent.turn_left()
elif action == 3:
    agent.turn_right()
```

---

## Troubleshooting

### No trained model found
```bash
# Make sure you've trained the model
cd src/navigation
python train_low_level_nav.py
```

### Scene not found
```bash
# Check if scene exists
ls habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb

# If missing, download Habitat test scenes
cd habitat-sim
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes
```

### Display issues
```bash
# Use headless mode and save video
python test_nav.py --no-display --save-video
```

### Import errors
```bash
# Activate environment
conda activate habitat-lab-env

# Install missing packages
pip install opencv-python numpy-quaternion
```

---

## Next Steps

1. **Run basic visualization**: `python test_nav.py`
2. **Try examples**: `python navigation/example_usage.py --example 1`
3. **Read full guide**: `src/navigation/VISUALIZATION_GUIDE.md`
4. **Integrate into your HRL system**: See examples in `example_usage.py`

---

## Quick Reference: Action Mapping

| Model Output | Action Name | Habitat-Sim Call | Effect |
|--------------|-------------|------------------|--------|
| 0 | NO-OP | *(none)* | Do nothing |
| 1 | MOVE_FORWARD | `agent.act("move_forward")` | Move 0.5m forward |
| 2 | TURN_LEFT | `agent.act("turn_left")` | Rotate 10° left |
| 3 | TURN_RIGHT | `agent.act("turn_right")` | Rotate 10° right |

---

## Questions?

Check the detailed guide: `src/navigation/VISUALIZATION_GUIDE.md`
