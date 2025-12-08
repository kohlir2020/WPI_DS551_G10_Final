"""
TensorBoard Metrics Explained
Understanding RL Training Metrics
"""

import os

# Create markdown explanation
explanation = """
# 📊 TensorBoard Metrics Explained

## Key Metrics You'll See

### 1. **rollout/ep_len_mean** (Episode Length Mean)
**What it measures:** Average number of steps per episode

**Definition:**
- ep_len = number of environment steps before episode ends
- mean = average across multiple episodes evaluated

**Interpretation:**
- ✅ Increasing → Agent learning to solve task longer (exploring more)
- ⚠️ Constant → Agent stuck in same behavior pattern
- ⚠️ Decreasing → Agent failing faster (worse performance)

**For Arm Reaching Task:**
- Max steps = 200
- If ep_len_mean = 150 → Agent survives 150 steps before success/failure
- Higher is better (exploring more, trying longer)

**Example:**
```
Iteration 1:  ep_len_mean = 50   (failing quickly)
Iteration 10: ep_len_mean = 180  (learning - solving longer!)
```

---

### 2. **rollout/ep_rew_mean** (Episode Reward Mean)
**What it measures:** Average total reward per episode

**Definition:**
- ep_rew = sum of all rewards in one episode
- mean = average across multiple episodes

**Interpretation:**
- ✅ Increasing → Agent performing better
- ✅ Plateauing high → Converged to good policy
- ⚠️ Decreasing → Learning degrading
- ⚠️ Unstable (wild swings) → Training is chaotic

**For Arm Reaching Task:**
- Base reward = 1/(1+distance_to_goal)  [0 to 1 per step]
- Bonus = +5 for reaching goal
- Max possible ≈ 200 + 5 = 205

**Example:**
```
Iteration 1:  ep_rew_mean = 50   (very far from goal)
Iteration 50: ep_rew_mean = 150  (getting closer!)
Iteration 100: ep_rew_mean = 180 (converged)
```

---

## 📈 **What Good Training Looks Like**

### PPO (Policy Gradient)
```
ep_rew_mean:    ↗️ Smooth upward curve
ep_len_mean:    ↗️ Gradually increasing
Stability:      🟢 Stable, consistent improvements
Speed:          🟢 Fast training
```
- Best for: Continuous control tasks
- Expected: Clean learning curve

### A2C (Actor-Critic, Synchronous)
```
ep_rew_mean:    ↗️ Smooth but might be slower
ep_len_mean:    ↗️ Steady increase
Stability:      🟡 More stable than PPO at times
Speed:          🟡 Medium speed
```
- Best for: Balanced stability/speed
- Expected: Similar to PPO but smoother

### SAC (Soft Actor-Critic)
```
ep_rew_mean:    ↗️ May plateau earlier
ep_len_mean:    ↗️ Stable increase
Stability:      🟢 Very stable (entropy regularization)
Speed:          🟡 Slower convergence
```
- Best for: Exploration and stability
- Expected: Consistent but may plateau sooner

---

## 🔍 **How to Interpret Comparison**

### Comparing in TensorBoard:
1. Go to **Scalars** tab
2. Select `rollout/ep_rew_mean` 
3. Compare all 3 algorithms:
   - PPO line (blue)
   - A2C line (orange)
   - SAC line (green)

### What to Look For:

**Winner Criteria:**
```
1. Highest final reward       → Best performance
2. Smoothest learning curve   → Most stable
3. Fastest convergence        → Fewest steps to learn
4. Least variance            → Most reliable
```

---

## 📊 **Expected Results for Your Task**

| Metric | PPO | A2C | SAC |
|--------|-----|-----|-----|
| Final Reward | ~180-200 | ~170-190 | ~160-180 |
| Convergence Speed | Fast (50k) | Medium (50k) | Slow (50k) |
| Stability | Good | Good | Excellent |
| Sample Efficiency | Best | Good | Worst |

---

## 💡 **Reading the Graphs**

### Good Training Curve:
```
Reward
   ^
200|                    ╱╱╱
180|                ╱╱╱
160|            ╱╱╱
140|        ╱╱╱
120|    ╱╱╱
100|╱╱╱
   +──────────────────────> Steps
   0      50k     100k
```
- Smooth upward trend
- No sudden drops
- Converges to plateau

### Bad Training Curve:
```
Reward
   ^
200|╱╲    ╱╲╱╲    ╱╲    ╱
180|   ╲╱╲╱    ╲╱╲╱╲╱╲╱╲
160|   (chaotic, unstable)
   +──────────────────────> Steps
```
- Wild oscillations
- Frequent drops
- Not converging

---

## 🎯 **Action Items**

1. **View Comparison:**
   - Open http://localhost:6006
   - Look at `rollout/ep_rew_mean` for all 3 algorithms

2. **Identify Winner:**
   - Which has highest final reward?
   - Which converges fastest?
   - Which is most stable?

3. **Decision:**
   - Use best algorithm for next phase (vision-based training)
   - Or ensemble if all perform similarly

---

## 📝 **Other Useful Metrics**

### train/* (Training metrics)
- `loss` - How wrong the model is
- `entropy_loss` - Exploration regularization
- `approx_kl` - Policy change per update
- `clip_fraction` - % of gradients clipped (PPO only)

### time/* (Performance metrics)
- `fps` - Frames per second
- `time_elapsed` - Wall-clock training time
- `total_timesteps` - Cumulative environment steps

---

## ✅ **Summary**

**ep_len_mean = How long agent lasts per episode**
- Higher = Agent learning, exploring more actions

**ep_rew_mean = How much reward agent gets per episode**  
- Higher = Agent getting better at task
- Main success metric

**Both should increase together during training.**

If both increase smoothly → ✅ Good training
If one increases, other decreases → ⚠️ Problem
If both flat → ❌ Not learning
"""

print(explanation)

# Save to file
with open("TENSORBOARD_GUIDE.md", "w") as f:
    f.write(explanation)

print("\n✅ Saved to TENSORBOARD_GUIDE.md")
