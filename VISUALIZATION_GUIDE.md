# Arm Reaching Task Visualization Guide

## Overview

This guide explains how to set up and use the visualization tools for the HRL arm reaching task. We provide two complementary visualization systems:

1. **Report Generation** - Comprehensive training metrics and convergence analysis
2. **Episode Visualization** - Detailed arm trajectory and performance visualization

## Part 1: Report Generation

### Purpose
Generate comprehensive graphs and reports from training logs to analyze:
- Reward curves and convergence
- Success rates over time
- Final distance distributions
- Actor and critic loss convergence
- Subgoal achievement efficiency
- Exploration noise decay

### Setup

The report generator is self-contained and requires only standard scientific Python libraries:

```bash
pip install matplotlib numpy scipy
```

### Usage

Generate all reports from training logs:

```bash
python src/arm/generate_reports.py --log_dir logs --output_dir reports
```

### Arguments

- `--log_dir`: Directory containing episode training logs (default: `logs`)
  - Should contain files like `episode_1.json`, `episode_2.json`, etc.
- `--output_dir`: Output directory for generated reports (default: `reports`)

### Generated Graphs

The script generates 7 comprehensive graphs:

#### 1. **Episode Rewards** (`01_episode_rewards.png`)
- Raw episode rewards with moving average smoothing
- Shows learning progression and stability
- 10-episode moving window for trend visualization

#### 2. **Success Rate** (`02_success_rate.png`)
- Moving success rate over 20-episode windows
- Shows when agent achieves task success
- Percentage scale (0-100%)

#### 3. **Final Distances** (`03_final_distances.png`)
- Scatter plot of final distances per episode
- Histogram distribution of final distances
- Red line indicates 0.3m success threshold

#### 4. **Loss Convergence** (`04_loss_convergence.png`)
- Separate actor and critic loss curves
- Raw values with 10-episode smoothing
- Histograms showing loss distributions
- Indicates learning stability

#### 5. **Subgoal Achievement** (`05_subgoal_success.png`)
- Bar chart of subgoals achieved per episode
- Moving average trend line (20-episode window)
- Shows high-level policy efficiency

#### 6. **Exploration Noise Decay** (`06_exploration_noise.png`)
- Actual exploration noise schedule
- Theoretical decay curve (500-episode decay)
- Validates noise decay parameter tuning

#### 7. **Combined Dashboard** (`07_combined_metrics.png`)
- 9-panel comprehensive summary
- All key metrics in single figure
- Includes statistics table
- Perfect for reports and presentations

### Example Workflow

```bash
# After training for 150 episodes
python src/arm/generate_reports.py --log_dir logs --output_dir my_reports

# View generated files
ls my_reports/
# Output: 01_episode_rewards.png 02_success_rate.png ... 07_combined_metrics.png
```

## Part 2: Episode Visualization

### Purpose
Visualize individual episodes to understand:
- Arm trajectory in 3D space
- Progress toward goal over time
- Action sequences and joint angles
- Success vs failure comparisons

### Setup

Requirements:
```bash
pip install matplotlib numpy
```

Optional (for full habitat support):
```bash
# This is already installed in your habitat environment
```

### Usage

Visualize episodes from training:

```bash
python src/arm/visualize_arm_episodes.py \
    --log_dir logs/arm_episodes \
    --output_dir visualizations \
    --num_episodes 10
```

### Arguments

- `--log_dir`: Directory containing episode log files (default: `logs/arm_episodes`)
  - Episode files should contain trajectory data
- `--output_dir`: Output directory for visualizations (default: `visualizations`)
- `--num_episodes`: Number of episodes to visualize (default: 5)

### Generated Visualizations

For each episode, generates:

#### 1. **2D Trajectories** (`ep001_trajectory_2d.png`)
- Top-down view (XY plane)
- Front view (XZ plane)
- Shows start position, end position, goal location
- Success threshold circle (0.3m radius)

#### 2. **3D Trajectories** (`ep001_trajectory_3d.png`)
- Full 3D end-effector path
- Distance-to-goal progression over steps
- Goal sphere visualization (0.3m radius)
- Success threshold line

#### 3. **Episode Metrics** (`ep001_metrics.png`)
- 9-panel detailed analysis:
  - Cumulative reward curve
  - Step-by-step rewards
  - Distance progression
  - 3D action components (X, Y, Z)
  - Joint angle evolution (3 selected joints)

#### 4. **Success/Failure Comparison** (`success_comparison.png`)
- Average trajectories: successful vs failed
- Episode length distributions
- Success rate by distance threshold
- Statistical summary

### Example Workflow

```bash
# During training, save episode trajectories:
# The trainer will save to logs/arm_episodes/episode_*.json

# After training reaches milestone (e.g., 100 episodes):
python src/arm/visualize_arm_episodes.py \
    --log_dir logs/arm_episodes \
    --output_dir visualizations \
    --num_episodes 20

# View results
ls visualizations/
# Shows trajectory plots, metric plots, etc.
```

## Integration with Training

### Automatic Logging

The HRL trainer automatically logs episode data if you have JSON logging enabled. To ensure episode data is saved:

```python
# In training configuration
trainer.save_episode_trajectory = True
trainer.episode_log_dir = "logs/arm_episodes"
```

### Manual Episode Collection

If episodes aren't auto-logged, you can modify the trainer to collect trajectory data:

```python
# During episode rollout
episode_data = {
    'success': ep_success,
    'reward': ep_reward,
    'final_distance': final_main_dist,
    'ee_positions': [ee_pos_1, ee_pos_2, ...],
    'goal_position': self.main_goal,
    'distances_to_goal': [dist_1, dist_2, ...],
    'actions': [...],
    'rewards': [...],
    'joint_angles': [...]
}

# Save episode
with open(f"logs/arm_episodes/episode_{ep_num}.json", 'w') as f:
    json.dump(episode_data, f, default=float)
```

## Report Customization

### Adjusting Smoothing Windows

Edit `generate_reports.py` to change smoothing:

```python
# In TrainingReportGenerator methods:
# Default window=10 for rewards
self.plot_episode_rewards(window=20)  # Larger window = more smooth

# Default window=20 for success rate
self.plot_success_rate(window=30)
```

### Custom Time Ranges

To analyze specific episodes:

```python
# Modify _extract_metric to filter
episodes = sorted([k for k in self.episodes.keys() if k >= 100 and k <= 200])
```

### Output Format

All graphs are saved as high-resolution PNG (150 DPI):
- Good for presentations
- Compatible with LaTeX/PDF reports
- Suitable for printing

## Troubleshooting

### Issue: "No episodes loaded"
**Solution**: Check that log files exist in the specified directory:
```bash
ls logs/  # or your --log_dir
ls logs/episode_*.json  # Check for episode files
```

### Issue: Missing metrics in graphs
**Solution**: Ensure training code is logging all required metrics:
- `reward` - Total episode reward
- `success` - Boolean success flag
- `final_dist` - Final distance to goal
- `critic_loss`, `actor_loss` - Network losses (new in recent commit)
- `subgoal_successes` - Subgoals achieved
- `exploration_noise` - Current noise level

### Issue: Memory error with large datasets
**Solution**: Reduce `--num_episodes` or process in batches:
```bash
python src/arm/visualize_arm_episodes.py --log_dir logs/arm_episodes --num_episodes 5
```

## Best Practices

1. **Generate reports frequently** - After 50, 100, 150, 200, 300 episodes
2. **Compare runs** - Generate reports from different training runs for comparison
3. **Archive results** - Keep timestamped versions of reports
4. **Use for debugging** - Unusual patterns in loss or distance plots indicate issues

## Example: Complete Visualization Workflow

```bash
# 1. Train for some episodes
python src/arm/hac_continuous_her_arm.py --num_episodes 150

# 2. Generate training report
python src/arm/generate_reports.py --log_dir logs --output_dir reports/run1

# 3. Visualize best episodes
python src/arm/visualize_arm_episodes.py \
    --log_dir logs/arm_episodes \
    --output_dir visualizations/run1 \
    --num_episodes 10

# 4. Analyze results in reports/ and visualizations/

# 5. Continue training with improved parameters
python src/arm/hac_continuous_her_arm.py --num_episodes 200

# 6. Generate updated reports
python src/arm/generate_reports.py --log_dir logs --output_dir reports/run2
```

## Additional Resources

- **Matplotlib Documentation**: https://matplotlib.org/
- **NumPy Arrays**: https://numpy.org/doc/stable/reference/generated/numpy.array.html
- **JSON Logging**: Python's built-in `json` module
- **HRL Architecture**: See `src/arm/hac_continuous_her_arm.py`

## Contact & Support

For issues with visualization scripts:
1. Check troubleshooting section above
2. Verify file paths and permissions
3. Ensure all required packages are installed
4. Check that JSON/episode files are valid

---

**Last Updated**: 2024
**Version**: 1.0
