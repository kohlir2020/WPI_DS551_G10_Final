# Scene and Training Information

## Issue 1: Different Scenes for Different Skills

**YES**, you're correct - the skills were trained in different scenes:

### Navigation (HRL)
- **Scene**: Skokloster Castle (`habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`)
- **Training**: Both high-level and low-level navigation policies trained here
- **Navigable Area**:
  - X: [-9.76, 8.57] (18.33m wide)
  - Y: [0.0, 0.4] (floor level only)
  - Z: [0.97, 25.60] (24.62m deep)

### Arm Reaching
- **Scene**: Realistic simulation fallback (not actual Habitat scene)
- **Training**: PPO/A2C/SAC models trained with `use_habitat=False`
- **Environment**: `HabitatArmReachingEnv` with kinematic simulation
- **Note**: Originally intended for apartment_1 scene, but current models use fallback

## Issue 2: Navigable Areas in Skokloster Castle

Run `python explore_scene.py` to see actual navigable points.

### Safe Navigation Targets (Verified)
```python
# Use these coordinates - they're confirmed navigable:
safe_points = [
    [-4.0, 0.2, 13.5],   # Center area
    [4.1, 0.2, 6.6],     # East area
    [-1.5, 0.1, 20.3],   # South area
    [-4.3, 0.2, 12.0],   # West area
    [4.5, 0.2, 10.4],    # North area
    [6.2, 0.2, 5.3],     # Far east
]
```

### Why Robot Gets Stuck
- Tables and furniture create obstacles
- Your old coordinates (like `[8.0, 0.2, 5.0]` or `[12.0, 0.2, 8.0]`) are **not on navmesh**
- The LLM prompt had **made-up numbers** that aren't actually navigable
- **Fixed**: Updated LLM prompt with real coordinates from scene analysis

## Issue 3: Finding Scene Information

### Method 1: Use `explore_scene.py`
```bash
conda activate habitat-lab-env
python explore_scene.py
```

Shows:
- Navigable bounds
- 20 random navigable samples
- Tests specific coordinates
- Suggests good targets

### Method 2: In Code
```python
import habitat_sim

sim = habitat_sim.Simulator(cfg)
pathfinder = sim.pathfinder

# Get bounds
bounds = pathfinder.get_bounds()
print(f"Min: {bounds[0]}, Max: {bounds[1]}")

# Get random navigable points
for _ in range(10):
    point = pathfinder.get_random_navigable_point()
    print(point)

# Test if a point is navigable
point = [10.0, 0.2, 5.0]
snapped = pathfinder.snap_point(point)
if not np.isnan(snapped).any():
    print(f"Navigable! Snapped to: {snapped}")
else:
    print("Not navigable")
```

## Issue 4: Demo Mode for Presentations

### Usage
```bash
# Enable demo mode with 2-second pauses
python src/main.py --demo-mode

# Custom pause duration
python src/main.py --demo-mode --demo-pause 3.0

# With LLM and demo mode
python src/main.py --goal "navigate to center area" --use-llm --demo-mode
```

### What Demo Mode Shows
1. **Task-level info**: Which skill is being executed (from LLM plan)
2. **Model-level info**: Which specific model (high-level vs low-level PPO)
3. **Step-level info**: 
   - For HRL Navigation: Shows HIGH-LEVEL steps (each running 50 low-level steps)
   - For Arm: Shows which algorithm (PPO/A2C/SAC)
4. **Pauses**: Between tasks and every 10 steps

### Demo Output Example
```
🎯 EXECUTING TASK 1 ================================================
Skill: navigate
Parameters: {'target': [4.1, 0.2, 6.6]}
====================================================================

🎬 DEMO MODE: HRL_NAVIGATION ========================================
  This skill uses Hierarchical Reinforcement Learning (HRL):
  - High-level policy: Selects subgoals
  - Low-level policy: Executes primitive actions to reach subgoals
  - Each HIGH-LEVEL step runs up to 50 LOW-LEVEL steps
====================================================================

  🔄 HIGH-LEVEL STEP 0 (low-level steps 0-50)
     Model: High-level PPO selects next subgoal
  Step 10/500: pos=[-15.8, 0.2, 5.1], dist=20.45m (HL step 0)
  Step 20/500: pos=[-14.2, 0.2, 5.3], dist=19.12m (HL step 0)
  ...
  🔄 HIGH-LEVEL STEP 1 (low-level steps 50-100)
     Model: High-level PPO selects next subgoal
```

## Summary of Fixes Applied

1. ✅ **Fixed LLM prompt** with actual navigable coordinates from Skokloster Castle
2. ✅ **Fixed hardcoded plans** to use verified navigable points
3. ✅ **Added `explore_scene.py`** to analyze any scene's navigable areas
4. ✅ **Added demo mode** (`--demo-mode` flag) with detailed hierarchical visualization
5. ✅ **Documented scene mismatch** between navigation (Skokloster) and arm (simulation)

## Testing
```bash
# Test with actual navigable coordinates
conda activate habitat-lab-env
python src/main.py --demo-mode

# Explore scene first
python explore_scene.py

# Then run with LLM using real coordinates
export OPENAI_API_KEY='your-key'
python src/main.py --goal "move to the center area" --use-llm --demo-mode
```
