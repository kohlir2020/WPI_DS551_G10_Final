"""
Reward Function Update - Dense Reward Shaping
Based on IKEA Furniture Assembly paper
"""

explanation = """
# 🎯 **Improved Reward Function - Dense Reward Shaping**

## Problem with Original Reward
Your original reward was **too sparse**:
```python
reward = 1.0 / (1.0 + distance)  # Only 0-1 per step
+ 5.0 if success                 # Rare bonus
```

This meant:
- Agent gets almost no reward while learning
- A2C and SAC struggle because they need frequent signal
- PPO works better but still suboptimal

---

## New Reward Function - Paper-Based Approach

Inspired by IKEA Furniture Assembly paper (your reference):

```python
reward = 20 * I_success              # Success bonus
        + 20 * Δ_distance            # Dense reward for progress
        - 0.01                       # Step penalty for efficiency
        - collision_penalty          # (future: collision avoidance)
```

Where:
- **I_success** = 1 if reaching goal, 0 otherwise
- **Δ_distance** = prev_distance - current_distance (positive = moving closer)
- Step penalty = small cost per step to encourage efficiency

---

## Why This Works Better

### Original (Sparse):
```
Step 1:  reward = 0.33  (too small)
Step 2:  reward = 0.40
Step 3:  reward = 0.35
...
Step 100: reward = 5.0  (finally!)

Agent barely learns because signals are weak
```

### New (Dense):
```
Step 1:  distance = 2.0m → 1.8m: reward = 20*(0.2) = +4.0 ✅ STRONG SIGNAL
Step 2:  distance = 1.8m → 1.6m: reward = 20*(0.2) = +4.0 ✅ REWARDS PROGRESS
Step 3:  distance = 1.6m → 1.4m: reward = 20*(0.2) = +4.0 ✅ CONTINUOUS LEARNING
...
Step 100: distance = 0.1m → 0.0m: reward = 20.0 + 20*(0.1) = +22.0 ✅ SUCCESS!

Agent gets constant learning signal throughout episode
```

---

## Key Components Explained

### 1. **Success Bonus: 20 points**
```python
if distance < 0.15:
    reward += 20.0
```
- Large enough to be significant
- Gives clear goal signal
- Not so large that it dominates

### 2. **Dense Reward: 20 × Δ_distance**
```python
delta_distance = prev_distance - current_distance
reward += 20.0 * delta_distance
```

This is the KEY change that helps A2C/SAC:
- Moving 0.1m closer → +2.0 reward
- Moving 0.5m closer → +10.0 reward
- Moving away → negative reward

**Why 20?** Matches paper's scale for arm reaching tasks

### 3. **Step Penalty: -0.01**
```python
reward -= 0.01
```
- Encourages agent to reach goal quickly
- Prevents infinite loops
- Not too harsh (doesn't overpower progress reward)

---

## Expected Results

### Before (Original Reward):
```
PPO:  ✅ Good (steep curve, learns fast)
A2C:  ⚠️ Flat (struggles with sparse reward)
SAC:  ⚠️ Flat (entropy prevents learning)
```

### After (Dense Reward):
```
PPO:  ✅✅ Excellent (faster, higher rewards)
A2C:  ✅ Good (dense signal helps)
SAC:  ✅ Good (frequent reward enables learning)
```

---

## How Dense Reward Helps Each Algorithm

### PPO (Policy Gradient)
- Already had some success
- Dense reward → faster convergence
- Expected: Reward increases 30-50% faster

### A2C (Actor-Critic, Sync)
- Critic network learns value function from rewards
- Sparse rewards → hard to learn value
- Dense rewards → critic has clear gradient
- Expected: Transforms from flat to steady increase

### SAC (Soft Actor-Critic)
- Entropy regularization explores more
- Entropy bonus isn't enough without good reward signal
- Dense rewards → balances exploration vs. exploitation
- Expected: Shows improvement after entropy kicks in

---

## Implementation Details

```python
def _get_reward(self, distance, prev_distance=None):
    """Dense reward with multiple components"""
    reward = 0.0
    
    # Success bonus
    if distance < 0.15:
        reward += 20.0
    
    # Dense progress reward (key for A2C/SAC)
    if prev_distance is not None:
        delta_distance = prev_distance - distance
        reward += 20.0 * delta_distance  # Scale: 20
    else:
        # First step fallback
        reward += 5.0 * (1.0 / (1.0 + distance))
    
    # Step penalty (efficiency)
    reward -= 0.01
    
    return reward
```

Key changes in environment:
1. Added `self.prev_distance` to track movement
2. Pass `prev_distance` to reward calculation
3. Reset `prev_distance` on episode start
4. Update `prev_distance` after each step

---

## Tuning Recommendations

If training still needs improvement:

### Too Slow?
- Increase scale: `20.0 * delta_distance` → `30.0 * delta_distance`
- Reduce step penalty: `0.01` → `0.001`

### Oscillating?
- Add velocity penalty: penalize large actions
- Increase step penalty slightly: `0.01` → `0.02`

### Too Conservative (not reaching goal)?
- Increase success bonus: `20.0` → `30.0`
- Or add intermediate bonus for getting close: `reward += 10.0 if distance < 0.5 else 0`

### Not Exploring Enough (SAC specific)?
- Entropy scale is already auto-tuned
- Can increase initial entropy: SAC(ent_coef="auto_0.1")

---

## Paper Reference

Your paper's reward structure for different skills:

**Pick Task:**
```
r_t = 20*I_success 
    + 5*I_pickup 
    + 20*Δ_o_arm * I_!holding    ← Dense reward for reaching
    + 20*Δ_r_arm * I_holding    ← Dense reward for returning
    - max(0.001*C_t, 1.0)        ← Collision penalty
    - 10*I_force
    - 5*I_wrong - 5*I_dropped
```

**Key insight:** Δ_o_arm (change in distance) multiplied by 20 is the main learning signal

Your updated function incorporates this principle.

---

## Monitoring in TensorBoard

After retraining, look for:

1. **Reward curves should increase (not flat):**
   ```
   Before: A2C/SAC flat around 20-30
   After:  A2C/SAC increasing to 150-180
   ```

2. **All 3 algorithms should show similar patterns:**
   ```
   PPO: Steepest
   A2C: Medium
   SAC: Smoothest
   ```

3. **No oscillation (wild swings):**
   - Should be smooth curves
   - If oscillating → reduce scale or add more averaging

---

## Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| Reward per step | 0.3-1.0 (sparse) | 0-20+ (dense) |
| A2C performance | Flat (bad) | Increasing (good) |
| SAC performance | Flat (bad) | Increasing (good) |
| Learning signal | Rare | Constant |
| Convergence speed | Medium | Fast |
| Algorithm balance | PPO dominates | More balanced |

---

## Next Steps

1. Wait for retraining to complete (~1-2 hours total)
2. View new results in TensorBoard
3. Verify all 3 algorithms now show improvement
4. Select best for vision-based training
5. Proceed to multi-camera observations

---

## Questions to Check

After new training:
- ✅ Are A2C/SAC curves now increasing (not flat)?
- ✅ Are all 3 algorithms reaching ~150-180 reward?
- ✅ Is learning smooth or noisy?
- ✅ Which algorithm converges fastest now?
"""

print(explanation)
