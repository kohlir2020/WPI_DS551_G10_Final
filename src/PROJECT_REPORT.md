# Hierarchical Reinforcement Learning for Robot Navigation
## CS551 Final Project - Team G10

---

## 1. PROJECT OVERVIEW

**Objective:** Develop a hierarchical reinforcement learning system for robot navigation in indoor environments using Habitat-Sim.

**Architecture:** Two-level hierarchy
- **Low-Level Controller:** Learns primitive navigation skills (move to nearby waypoint)
- **High-Level Manager:** Learns to select subgoals for long-range navigation

**Environment:** Habitat-Sim with Skokloster Castle scene

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Low-Level Navigation Agent

**Purpose:** Navigate to nearby goals (2-8 meters)

**Observation Space:**
- Distance to goal (0-50m)
- Relative angle to goal (-π to π radians)

**Action Space:** Discrete(4)
- 0: NO-OP
- 1: MOVE_FORWARD (0.5m)
- 2: TURN_LEFT (10°)
- 3: TURN_RIGHT (10°)

**Reward Function:**
```python
reward = 5.0 * progress              # Strong directional progress
reward -= 0.001 * distance           # Encourage staying close
reward -= 0.01                       # Time penalty
if distance < 0.5m:
    reward += 10.0                   # Success bonus
```

**Training Configuration:**
- Algorithm: PPO
- Total Timesteps: 250,000
- Learning Rate: 3e-4
- Entropy Coefficient: 0.005
- Episode Horizon: 150 steps
- Success Threshold: 0.5m

**Results:**
- **Success Rate: 90% (18/20 episodes)**
- Average Steps to Goal: 15.5
- Training Time: ~15 minutes (CPU)

---

### 2.2 High-Level Manager Agent

**Purpose:** Select intermediate subgoals for long-range navigation (8-20 meters)

**Observation Space:**
- Distance to main goal (0-100m)
- Relative angle to main goal (-π to π radians)

**Action Space:** Discrete(8)
- 8 directions on a circle (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)

**Subgoal Generation:**
- Radius: 5.0 meters
- Validation: Check navigability, path existence
- Fallbacks: Random navigable points if primary fails

**Reward Function:**
```python
reward = 10.0 * progress             # Strong progress toward main goal
if progress > 2.0m:
    reward += 5.0                    # Bonus for significant progress
if progress > 1.0m:
    reward += 2.0
if movement < 0.5m:
    reward -= 1.0                    # Penalty for getting stuck
reward -= 0.05                       # Small time penalty
if distance < 0.6m:
    reward += 50.0                   # Large success bonus
```

**Training Configuration:**
- Algorithm: PPO
- Total Timesteps: 1,000,000
- Learning Rate: 3e-4
- Entropy Coefficient: 0.08 (high exploration)
- n_steps: 512
- n_epochs: 10
- Option Horizon: 50 low-level steps per high-level action
- Episode Horizon: 20 high-level steps

**Results:**
- **Success Rate: 45% (9/20 episodes)**
- Average Steps to Goal: 8.3 (high-level)
- Average Final Distance (failures): 2.29m
- Training Time: ~90 minutes (CPU)

---

## 3. DEVELOPMENT PROCESS & CHALLENGES

### 3.1 Initial Attempts

**First Training Run:**
- Low-level: 100k timesteps
- Results: 0% success rate
- Issue: Insufficient training time, weak reward signal

**Diagnosis:** Agent showed some learning (rewards improved from -4.63 to -4.19) but couldn't reach goals consistently.

---

### 3.2 Low-Level Improvements

**Changes Made:**
1. Increased training to 250k timesteps
2. Implemented curriculum learning (goals between 2-8m)
3. Stronger reward shaping (5x progress multiplier)
4. Added distance-based bonuses (within 0.5m, 1m, 2m)

**Result:** Improved to 90% success rate ✓

---

### 3.3 High-Level Development - Major Challenge

#### Initial High-Level Training (100k timesteps)

**Results:** Complete failure
- Success rate: 0%
- Agent behavior: Always selected action 6 (policy collapse)
- Movement: Agent froze after 1-2 steps

#### Root Cause Analysis

**Diagnostic Testing Revealed:**
```
Test observations:
  [15m ahead]  → action 6 (55% probability)
  [15m right]  → action 6 (53% probability)
  [15m left]   → action 6 (51% probability)
  [15m behind] → action 6 (56% probability)
```

**Problem Identified:**
1. **Policy Collapse:** Agent ignored observations, always picked same action
2. **Insufficient Exploration:** ent_coef=0.01 too low
3. **Invalid Subgoals:** No validation led to unreachable targets
4. **Agent Got Stuck:** Low-level couldn't reach bad subgoals

**Why This Happened:**
- With low exploration, agent converged too quickly to suboptimal policy
- One early success with action 6 → kept repeating it
- Never explored other directions sufficiently
- Hierarchical RL requires MORE exploration than single-level

---

#### Solution Implementation

**Key Changes:**

1. **Increased Exploration (CRITICAL)**
   ```python
   ent_coef = 0.08  # Was 0.01 - 8x increase!
   ```
   Forces agent to try all 8 directions extensively

2. **Longer Training**
   ```python
   total_timesteps = 1_000_000  # Was 100_000
   ```

3. **Better Reward Shaping**
   ```python
   reward = 10.0 * progress  # Was 4.0 - stronger signal
   reward += 50.0 on success # Was 30.0 - bigger incentive
   ```

4. **Subgoal Validation**
   - Check if point is on navmesh
   - Verify path exists from agent to subgoal
   - Multiple fallback strategies
   - Minimum movement requirement (1.5m)

5. **More Updates Per Batch**
   ```python
   n_epochs = 10  # Was 4
   n_steps = 512  # Was 128
   ```

**Results After Improvements:**

Training at 500k timesteps:
- Success rate: 40% (2/5 quick eval)
- ep_rew_mean: improved to 96.6
- Agent showed diverse action selection

Training at 1M timesteps:
- **Success rate: 45% (9/20 episodes)**
- Average steps: 8.3
- Agent behavior: Picks different actions based on goal direction ✓
- All subgoals valid (100% validation rate)

---

## 4. TECHNICAL CHALLENGES & SOLUTIONS

### 4.1 GPU Compatibility Issues

**Problem:** RTX 5070 not compatible with Habitat-Sim CUDA version

**Solution:** Used CPU mode throughout
```python
device="cpu"
backend_cfg.gpu_device_id = -1
```

**Impact:** Slower training but acceptable (90 min for 1M steps)

---

### 4.2 Habitat-Sim Configuration

**Problem:** OmegaConf read-only config errors

**Solution:**
```python
from omegaconf import OmegaConf
OmegaConf.set_struct(config, False)  # Unlock
# Make modifications
OmegaConf.set_struct(config, True)   # Lock again
```

---

### 4.3 Observation Calculation Issues

**Problem:** NaN values in angle calculations when agent/goal overlap

**Solution:** Added safety checks
```python
if distance < 1e-6:
    return np.array([0.0, 0.0])
if np.isnan(angle):
    angle = 0.0
```

---

### 4.4 Subgoal Generation Failures

**Problem:** `snap_point()` returned NaN or points too close to agent

**Solution:** Multi-tier fallback system
```python
# Try multiple distances
for dist in [5.0, 3.5, 2.5]:
    # Try snapping
    if valid and movement > 1.5m:
        return subgoal
# Fallback 1: Random navigable point
# Fallback 2: Move toward main goal
# Fallback 3: Minimal movement
```

---

## 5. KEY INSIGHTS & LESSONS LEARNED

### 5.1 Hierarchical RL is Harder Than Single-Level

**Why:**
- Credit assignment problem (which subgoal choice led to success?)
- Compound errors (bad high-level choice → low-level can't recover)
- Sparse rewards (main goal far away)
- Need much more exploration

**Evidence:**
- Low-level: 90% success with 250k timesteps
- High-level: 45% success with 1M timesteps (4x more training!)

---

### 5.2 Exploration is Critical

**Lesson:** entropy_coef is the most important hyperparameter for HRL

**Our Experience:**
- ent_coef=0.01: Complete policy collapse (0% success)
- ent_coef=0.08: System works (45% success)

**Why:** High-level needs to try ALL directions many times to learn which work best for different situations.

---

### 5.3 Reward Shaping Matters

**Key Principle:** Rewards must be proportional to task difficulty

Low-level (easy task, nearby goals):
```python
reward = 5.0 * progress + 10.0 * success
```

High-level (hard task, far goals):
```python
reward = 10.0 * progress + 50.0 * success  # Stronger signals!
```

---

### 5.4 Validation is Essential

**Lesson:** Always validate generated targets in robotics

Without validation:
- Agent selects action 6
- Generates invalid subgoal (NaN or unreachable)
- Low-level tries to navigate → fails
- Agent stuck, no learning

With validation:
- 100% valid subgoals (13/13 in evaluation)
- Consistent movement every step
- Agent actually reaches goals

---

## 6. RESULTS COMPARISON

| Metric | Low-Level | High-Level (HRL) |
|--------|-----------|------------------|
| Success Rate | 90% | 45% |
| Avg Steps to Goal | 15.5 | 8.3 |
| Training Time | 15 min | 90 min |
| Training Steps | 250k | 1M |
| Task Complexity | Simple (2-8m) | Complex (8-20m) |

**Key Observation:** When HRL succeeds, it's more efficient (8.3 vs 15.5 steps) because it plans at a higher level. But it's harder to train and less reliable.

---

## 7. DIAGNOSTIC METHODOLOGY

**Systematic Debugging Approach:**

1. **Created diagnostic script** to test each component:
   - Model loading
   - Environment functionality
   - Policy predictions
   - Actual execution

2. **Identified policy collapse** through action distribution analysis:
   ```
   All observations → Same action (clearly wrong!)
   ```

3. **Traced execution** to find where agent got stuck

4. **Tested fixes incrementally:**
   - First: Increase entropy → Actions diverse
   - Then: Validate subgoals → Movement consistent
   - Finally: Stronger rewards → Learning happens

**Lesson:** Don't just retrain hoping it works. Understand WHY it failed first.

---

## 8. PROJECT STRUCTURE

```
src/
├── simple_navigation_env.py          # Low-level environment
├── train_low_level_nav.py            # Low-level training script
├── evaluate_low_level_nav.py         # Low-level evaluation
├── hrl_highlevel_env.py              # High-level environment (original)
├── hrl_highlevel_env_improved.py     # High-level environment (fixed)
├── train_high_level.py               # High-level training (original)
├── train_high_level_improved.py      # High-level training (improved)
├── evaluate_high_level.py            # High-level evaluation
├── diagnose_hrl.py                   # Diagnostic tool

models/
├── lowlevel_curriculum_250k.zip      # Trained low-level agent (90%)
├── hl_improved/
│   └── highlevel_improved_final.zip  # Trained high-level agent (45%)
```

---

## 9. NEXT STEPS & IMPROVEMENTS

### 9.1 Immediate Improvements (Would Likely Work)

1. **Train Even Longer**
   - Try 2M timesteps for high-level
   - Expected: 50-55% success rate

2. **Curriculum Learning for High-Level**
   - Start with closer main goals (5-10m)
   - Gradually increase to 15-20m
   - Expected: Faster convergence, higher success

3. **Better Action Space**
   - Add "move directly toward goal" as action 0
   - Current 8 directions might miss optimal paths
   - Expected: More efficient navigation

4. **Adaptive Subgoal Distance**
   - Use 3m subgoals when close to goal
   - Use 7m subgoals when far from goal
   - Expected: Better fine-grained control

---

### 9.2 Advanced Improvements (Research Ideas)

1. **Options Framework**
   - Train multiple low-level skills (avoid obstacle, follow wall, etc.)
   - High-level selects which skill to use
   - Expected: More robust behavior

2. **Intrinsic Motivation**
   - Add curiosity bonus for exploring new areas
   - Helps with sparse reward problem
   - Expected: Better exploration of large environments

3. **Hindsight Experience Replay (HER)**
   - Relabel failed episodes as successes to closer goals
   - Improves sample efficiency
   - Expected: Faster training, higher success rate

4. **Multiple Scenes**
   - Train on various Habitat scenes
   - Test generalization to new environments
   - Expected: More robust, generalizable agent

5. **Visual Observations**
   - Replace [distance, angle] with RGB-D images
   - More realistic but much harder to train
   - Expected: Better sim-to-real transfer potential

6. **Dynamic Obstacles**
   - Add moving obstacles to environment
   - Requires reactive planning
   - Expected: More realistic navigation system

---

### 9.3 Alternative Approaches to Try

1. **Feudal Networks**
   - High-level sets direction gradients
   - Low-level maximizes movement in that direction
   - Potentially easier credit assignment

2. **Model-Based Planning**
   - Learn world model
   - Plan paths explicitly
   - Combine with RL for control

3. **Graph-Based Navigation**
   - Pre-compute navigation graph
   - High-level selects waypoints on graph
   - More structured than free subgoal selection

---

## 10. REPRODUCTION INSTRUCTIONS

### Environment Setup
```bash
# Create conda environment
conda create -n habitat python=3.9
conda activate habitat

# Install dependencies
pip install habitat-sim habitat-lab
pip install stable-baselines3[extra]
pip install numpy-quaternion

# Download Habitat test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes
```

### Training Low-Level Agent
```bash
cd src
python train_low_level_nav.py
# Expected: ~15 minutes, 90% success rate
```

### Training High-Level Agent
```bash
python train_high_level_improved.py
# Expected: ~90 minutes, 40-45% success rate
```

### Evaluation
```bash
# Evaluate low-level
python evaluate_low_level_nav.py

# Evaluate high-level
python evaluate_hl_corrected.py
```

---

## 11. REFERENCES & RESOURCES

**Papers:**
- Dayan & Hinton (1993): Feudal Reinforcement Learning
- Vezhnevets et al. (2017): FeUdal Networks for Hierarchical RL
- Nachum et al. (2018): Data-Efficient Hierarchical RL

**Libraries:**
- Habitat-Sim: https://github.com/facebookresearch/habitat-sim
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

**Key Concepts:**
- Hierarchical RL
- Options Framework
- Temporal Abstraction
- Credit Assignment Problem
- Exploration vs Exploitation

---

## 12. TEAM CONTRIBUTIONS & ACKNOWLEDGMENTS

**Development Process:**
- Environment setup and integration
- Low-level navigation system implementation
- High-level hierarchical system development
- Extensive debugging and diagnostic tool creation
- Hyperparameter tuning and optimization

**Key Insight:** The debugging phase (diagnosing policy collapse) was as valuable as the initial implementation. Understanding failure modes is critical in RL research.

---

## APPENDIX A: HYPERPARAMETER TABLES

### Low-Level Agent
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 3e-4 | Standard for PPO |
| n_steps | 512 | Balance batch size/compute |
| batch_size | 128 | Stable gradients |
| gamma | 0.90 | Shorter horizon task |
| ent_coef | 0.005 | Some exploration needed |
| n_epochs | 4 | Standard |
| episode_horizon | 150 | ~75m max travel |
| success_threshold | 0.5m | Reasonable precision |

### High-Level Agent (Final)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 3e-4 | Standard for PPO |
| n_steps | 512 | More samples per update |
| batch_size | 64 | Smaller for better gradients |
| gamma | 0.98 | Long-term planning |
| ent_coef | 0.08 | HIGH exploration (critical!) |
| n_epochs | 10 | Learn more per batch |
| subgoal_distance | 5.0m | Achievable by low-level |
| option_horizon | 50 | Time to reach subgoal |
| episode_horizon | 20 | Max high-level steps |

---

## APPENDIX B: COMMON ERRORS & SOLUTIONS

**Error: "snap_point returned NaN"**
- Solution: Add fallback to random navigable point

**Error: "Policy always picks same action"**
- Solution: Increase entropy coefficient (0.05-0.10)

**Error: "Agent doesn't move"**
- Solution: Validate subgoals, check if low-level model loaded correctly

**Error: "Rewards not improving"**
- Solution: Check reward scale, ensure progress is measurable

**Error: "CUDA initialization failed"**
- Solution: Use CPU mode (gpu_device_id = -1)

---

## CONCLUSION

This project successfully implemented a hierarchical reinforcement learning system for robot navigation, achieving:
- 90% success rate for low-level navigation
- 45% success rate for hierarchical navigation (10-20m goals)
- Systematic debugging methodology
- Key insights into HRL challenges

The main challenge was **policy collapse in the high-level agent**, which required:
- 8x increase in exploration coefficient
- 10x longer training time
- Improved reward shaping
- Robust subgoal validation

This demonstrates that hierarchical RL requires careful tuning and is significantly more challenging than single-level RL, but offers benefits in terms of planning efficiency when successful.

**Key Takeaway:** In hierarchical RL, exploration is paramount. Without sufficient exploration (high entropy), the high-level policy will collapse to a single action and fail to learn.
