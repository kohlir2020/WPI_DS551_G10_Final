# ARM REACHING ENVIRONMENT - IMPLEMENTATION GUIDE

## Overview
Created new **Fetch Arm Reaching Environment** for training the robot arm to reach toward objects (drawer/fridge handles) with automatic gripper trigger at 15cm.

## What Changed

### New Files Created

#### 1. `src/arm/fetch_arm_reaching_env.py`
Gymnasium-compatible environment for arm reaching task.

**Key Features:**
- **Observation Space**: `[distance_to_goal, arm_joint_angles(7)]`
  - Distance: 0-5m to goal position
  - Arm angles: 7 joint angles (-π to π radians)
  
- **Action Space**: Continuous arm joint velocities (7-DOF)
  - Each joint: -1.0 to +1.0 rad/s
  - Applied with 50ms timestep for smooth movement
  
- **Goal Setup**:
  - Random x,y position within 1m of base
  - Fixed height = drawer/fridge handle height (0.5m default)
  - Success = gripper within 15cm of goal (auto-triggers gripper)

- **Reward Structure**:
  - +1.0 when reaching goal (success)
  - -0.01 * distance for proximity reward (learns to approach goal)

- **Episode Duration**: 200 steps max (~10 seconds at 50ms/step)

#### 2. `src/arm/train_fetch_arm.py`
PPO trainer for the arm reaching task.

**Training Configuration:**
```python
Learning rate: 3e-4
PPO N-steps: 512
Batch size: 64
Epochs per update: 10
Entropy coefficient: 0.01 (encourages exploration)
Default total steps: 100,000
```

**Features:**
- Checkpoint saving every 10k steps
- Evaluation episodes every 5k steps
- TensorBoard logging for training curves
- Automatic best model selection

**Usage:**
```bash
python src/arm/train_fetch_arm.py --steps 100000 --lr 3e-4 --batch-size 64
```

### Modified Files

#### `launch_parallel_training.py`
Updated to use new arm trainer:
- Changed reference from `train_fetch_nav.py` → `train_fetch_arm.py`
- Updated process names from 'fetch' → 'fetch_arm'
- Updated logging for clarity

## Task Design Details

### Why Not Full Gripper Control?
The task assumes:
1. ✓ Robot base already navigated to correct position
2. ✓ Arm reaches toward goal (what we train)
3. ✓ Gripper triggers automatically at 15cm (distance-based rule, no learning)

**Benefits:**
- Simpler task (only reach, not manipulate)
- Automatic gripper prevents grip failures
- Fewer degrees of freedom to learn

### Arm Kinematics
**7-DOF Fetch Arm:**
```
shoulder_pan_joint      - Base rotation
shoulder_lift_joint     - Upper arm lift
upperarm_roll_joint     - Upper arm rotation
elbow_flex_joint        - Elbow bend
forearm_roll_joint      - Forearm rotation
wrist_flex_joint        - Wrist bend
wrist_roll_joint        - Wrist rotation
```

**Default Pose** (neutral position):
```
[0.0, 1.3, 0.0, -2.2, 0.0, 2.0, 0.0] radians
```

## Environment Details

### Class: `FetchArmReachingEnv`

#### Parameters
```python
scene_path: str
    Path to scene file (auto-detected from ../data/)

goal_height: float = 0.5
    Height of goal (drawer/fridge) in meters

success_distance: float = 0.15
    Distance threshold for gripper trigger (15cm)

max_steps: int = 200
    Maximum steps per episode
```

#### Methods

**`reset(seed=None, options=None)`**
- Samples random goal position (x,y,z)
- Resets arm to default neutral pose
- Returns initial observation

**`step(action)`**
- Takes 7-dim continuous action (joint velocities)
- Clips to [-1, 1] rad/s
- Returns: obs, reward, done, truncated, info

**`_get_obs()`**
- Computes distance to goal from gripper position
- Returns [distance, arm_angles[0:7]]

### Reward Function
```python
if distance < 0.15m:
    reward = +1.0  (success!)
else:
    reward = -0.01 * distance  (proximity bonus)
```

Episode terminates on:
- Success: distance < 0.15m
- Timeout: step >= 200

## Running Training

### Option 1: Docker (Recommended - all deps included)
```bash
docker-compose up -d
docker exec rl_training python src/arm/train_fetch_arm.py --steps 100000
```

### Option 2: Local (requires Habitat 2.0 + dependencies)
```bash
python src/arm/train_fetch_arm.py --steps 100000
```

### Option 3: Parallel Training (Nav + Arm)
```bash
python launch_parallel_training.py --mode both
```

Starts both:
- Navigation training (src/train_low_level_nav.py)
- Arm reaching training (src/arm/train_fetch_arm.py)

Logs saved to: `logs/parallel/`

## Training Output

### Logs Directory
```
logs/fetch_arm/{timestamp}/
├── checkpoints/
│   ├── arm_reaching_100000_steps.zip
│   ├── arm_reaching_200000_steps.zip
│   └── ...
├── best_model.zip
├── events.out.tfevents...
└── evaluations.npz
```

### TensorBoard Monitoring
```bash
tensorboard --logdir logs/fetch_arm/{timestamp}
```

### Model Saving
- Checkpoint: Every 10k steps
- Best model: When eval performance improves
- Final model: `{timestamp}/fetch_arm_reaching_final.zip`

## Integration with Skill Executor

The trained arm model will integrate into `skill_executor.py`:

```python
class SkillExecutor:
    def __init__(self):
        self.nav_skill = NavigationSkill(model="models/nav_model.zip")
        self.arm_skill = ArmSkill(model="models/arm_model.zip")  # NEW
    
    def execute_drawer_opening(self):
        # Step 1: Navigate to drawer
        nav_state = self.nav_skill.execute(goal_location)
        
        # Step 2: Reach toward drawer handle  
        arm_state = self.arm_skill.execute(goal_position)  # NEW
        
        # Step 3: Gripper triggers automatically at 15cm
        if arm_state['distance'] < 0.15:
            trigger_gripper()
```

## Expected Performance

**Navigation (Reference - Already Working):**
- Training time: ~3 minutes (250k steps)
- Success rate: ~80%+ by end of training

**Arm Reaching (New):**
- Training time: ~10-15 minutes (100k steps)
- Expected success rate: ~60-80% by end
- Can extend to 200k+ steps for higher accuracy

## Troubleshooting

### Graphics/EGL Error
```
Platform::WindowlessEglApplication::tryCreateContext(): cannot initialize EGL
```
**Solution**: Expected in container without GPU/display. Set:
```python
backend_cfg.enable_physics = True  # Already set
backend_cfg.gpu_device_id = -1     # CPU mode - already set
```
Graphics are disabled since we only need state observations, not images.

### Out of Memory (OOM)
If training crashes with memory error:
```bash
python src/arm/train_fetch_arm.py --steps 100000 --batch-size 32  # Reduce batch size
```

### Slow Training
GPU in Docker:
```bash
docker run --gpus all -v $(pwd):/workspace ... \
  python src/arm/train_fetch_arm.py
```

## Next Steps

1. ✓ Create arm reaching environment (DONE)
2. ✓ Create arm trainer (DONE)
3. ✓ Update parallel launcher (DONE)
4. **TODO**: Test arm environment in Docker
5. **TODO**: Train arm model (100k-200k steps)
6. **TODO**: Integrate with skill executor
7. **TODO**: Test combined nav+arm sequence

## Files Modified/Created

**Created:**
- `src/arm/fetch_arm_reaching_env.py` - Arm environment
- `src/arm/train_fetch_arm.py` - Arm trainer
- `ARM_REACHING_GUIDE.md` - This file

**Modified:**
- `launch_parallel_training.py` - Updated to use arm trainer

**No changes needed to:**
- `src/simple_navigation_env.py` - Still valid
- `src/train_low_level_nav.py` - Still valid
- `src/skill_executor.py` - Ready for arm integration
- Docker/docker-compose files - No changes needed
