# Answers to Your Questions

## Q1: Are arm and navigation trained in different scenes?

**YES**

- **Navigation (HRL)**: Trained in **Skokloster Castle** scene
  - Scene file: `habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb`
  - Both high-level and low-level PPO models trained here
  
- **Arm Reaching**: Trained in **realistic simulation fallback** (not actual scene)
  - Uses `HabitatArmReachingEnv` with `use_habitat=False`
  - Kinematic simulation, not Skokloster Castle
  - Models: PPO/A2C/SAC in `logs/simple_arm/`

## Q2: How to know which areas are navigable?

**Run the scene explorer:**
```bash
conda activate habitat-lab-env
python explore_scene.py
```

**Key findings for Skokloster Castle:**
- Navigable bounds: X[-9.76 to 8.57], Y[0.0-0.4], Z[0.97-25.60]
- Tables/furniture create obstacles
- **Safe coordinates** (verified):
  ```python
  [-4.0, 0.2, 13.5]   # Center
  [4.1, 0.2, 6.6]     # East
  [-1.5, 0.1, 20.3]   # South
  [-4.3, 0.2, 12.0]   # West
  [4.5, 0.2, 10.4]    # North
  ```

**Why robot got stuck:** Your old coordinates like `[8.0, 0.2, 5.0]` or `[12.0, 0.2, 8.0]` are **not on navmesh** - they hit tables/walls.

## Q3: Did LLM prompt use made-up numbers?

**YES - but FIXED now**

**Before:**
```python
# Made-up coordinates (not navigable!)
"Kitchen area: around [8.0, 0.2, 5.0]"
"Dining room: around [12.0, 0.2, 8.0]"
```

**After:**
```python
# Real coordinates from scene analysis
"Navigable area center: around [-4.0, 0.2, 13.5] or [4.1, 0.2, 6.6]"
"North area: [4.5, 0.2, 10.4]"
"South area: [-1.5, 0.1, 20.3]"
```

**How to find info for any scene:**
1. Use `explore_scene.py` (I created this for you)
2. Or in code:
   ```python
   pathfinder = sim.pathfinder
   bounds = pathfinder.get_bounds()
   samples = [pathfinder.get_random_navigable_point() for _ in range(20)]
   ```

## Q4: How to add demo mode for presentations?

**IMPLEMENTED - Use `--demo-mode` flag**

```bash
# Basic demo mode (2 second pauses)
python src/main.py --demo-mode

# Custom pause duration
python src/main.py --demo-mode --demo-pause 3.0

# With LLM planning
python src/main.py --goal "move to center" --use-llm --demo-mode
```

**What it shows:**

1. **Task level**: Which skill from LLM plan is executing
2. **Model level**: Which specific model (high-level vs low-level)
3. **Step level**: 
   - For HRL: Shows HIGH-LEVEL steps (each = 50 low-level steps)
   - For Arm: Shows algorithm (PPO/A2C/SAC)
4. **Pauses**: Automatic pauses to see what's happening

**Example output:**
```
🎯 EXECUTING TASK 1 ============================================
Skill: navigate
Parameters: {'target': [-4.0, 0.2, 13.5]}

🎬 DEMO MODE: HRL_NAVIGATION ===================================
  This skill uses Hierarchical Reinforcement Learning (HRL):
  - High-level policy: Selects subgoals
  - Low-level policy: Executes primitive actions
  - Each HIGH-LEVEL step runs up to 50 LOW-LEVEL steps

  🔄 HIGH-LEVEL STEP 0 (low-level steps 0-50)
     Model: High-level PPO selects next subgoal
  Step 10/500: pos=[-15.8, 0.2, 5.1], dist=20.45m (HL step 0)
  
  🔄 HIGH-LEVEL STEP 1 (low-level steps 50-100)
     Model: High-level PPO selects next subgoal
  Step 60/500: pos=[-14.2, 0.2, 5.3], dist=19.12m (HL step 1)
```

## All Fixes Applied

✅ **Fixed scene coordinates** - Updated LLM prompt and hardcoded plans with real navigable points  
✅ **Created scene explorer** - `explore_scene.py` to analyze any scene  
✅ **Added demo mode** - `--demo-mode` flag with hierarchical visualization  
✅ **Documented scenes** - Clarified navigation vs arm training scenes  

## Quick Reference

```bash
# Explore scene first
python explore_scene.py

# Run with demo mode
conda activate habitat-lab-env
python src/main.py --demo-mode

# With LLM (set API key first)
export OPENAI_API_KEY='your-key'
python src/main.py --goal "navigate to north area and pick up object" --use-llm --demo-mode
```
