# Getting Started - Parallel Training & Docker Setup

## ⚠️ CRITICAL: Path Issues Fixed (Dec 1, 2025)

**If you tried running training before and got "Scene not found" errors**, the issues have been fixed:

✅ All hardcoded scene paths now use relative paths to local `data/` directory
✅ Import paths fixed for training scripts
✅ Removed missing dependencies (check_env, tensorboard logging)

**Training now works!** Run:
```bash
python launch_parallel_training.py --mode both
```

For details, see `FIXES_APPLIED.md`

---

This guide walks you through the new parallel training and Docker containerization features.

## Quick Links
- **Read This First:** This file
- **What Was Fixed:** See `FIXES_APPLIED.md` (if training previously failed)
- **Architecture Overview:** See `.github/copilot-instructions.md`
- **Full Implementation Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Docker Troubleshooting:** See `DOCKER.md`

## What's New

### 1. **Parallel Trainer for Navigation + Fetch**
Train both navigation and fetch base simultaneously instead of sequentially.

**Before (Sequential):**
```
Navigation Training: ~15 min (250k steps)
    ↓ (wait)
Fetch Training: ~15 min (250k steps)3
*
─────────────────────────────
Total: ~30 minutes
```

**Now (Parallel):**
```
Navigation Training: ~15 min ┐
Fetch Training: ~15 min      ├─ Simultaneously
─────────────────────────────
Total: ~15 minutes (50% faster!)
```

### 2. **Docker Container with GPU Support**
Train inside a containerized environment with automatic GPU detection and all dependencies pre-installed.

### 3. **Skill Combination Framework**
Test how individual trained skills work together before implementing high-level HRL.

## Get Started (Choose One)

### Option A: Local Development (Fastest for Experimentation)

```bash
# 1. Navigate to project
cd ~/RL_final/WPI_DS551_G10_Final

# 2. Train both skills in parallel
python launch_parallel_training.py --mode both

# 3. Monitor in another terminal
tail -f logs/parallel/nav_training_*.log
tail -f logs/parallel/fetch_training_*.log

# 4. When done (~15 min), test skill combination
python src/skill_executor.py
```

**Logs:** `logs/parallel/nav_training_YYYYMMDD_HHMMSS.log`

### Option B: Docker (Recommended for HPC/Turing)

```bash
# 1. Navigate to project
cd ~/RL_final/WPI_DS551_G10_Final

# 2. Build Docker image (one-time, ~5 min)
docker build -t hrl-training:latest .

# 3. Start training container
docker-compose up -d hrl-training

# 4. Enter container and run training
docker-compose exec hrl-training bash
# Inside container:
python launch_parallel_training.py --mode both

# 5. Monitor TensorBoard (from host)
# Open http://localhost:6006 in browser

# 6. Stop when done
docker-compose stop hrl-training
```

### Option C: Interactive Menu (Beginner-Friendly)

```bash
# 1. Navigate to project
cd ~/RL_final/WPI_DS551_G10_Final

# 2. Run interactive launcher
./dev.sh

# 3. Select option 5 for parallel training
```

## Understanding the Output

### Parallel Training Output

```
========================================================
HIERARCHICAL RL - PARALLEL TRAINING LAUNCHER
Started: 2024-01-15 10:30:45.123456
========================================================

=========================================================
Starting Navigation Training
=========================================================
Output: logs/parallel/nav_training_20240115_103045.log
✓ Navigation training started (PID: 12345)

=========================================================
Starting Fetch Navigation Training
=========================================================
Output: logs/parallel/fetch_training_20240115_103045.log
✓ Fetch training started (PID: 12346)

===================================================
Monitoring Training Processes
===================================================

✓ NAVIGATION       | PID:  12345 | Running
✓ FETCH            | PID:  12346 | Running

✓ NAVIGATION       | Completed | Log: logs/parallel/nav_training_20240115_103045.log
✓ FETCH            | Completed | Log: logs/parallel/fetch_training_20240115_103045.log
```

### Skill Executor Output

```
======================================================================
EXECUTING SKILL SEQUENCE: ['navigation', 'fetch_navigation']
======================================================================

>>> Executing navigation...
Step   0: FWD   | dist=  8.32m | reward=  0.07
Step   1: LEFT  | dist=  8.01m | reward=  0.03
...
✓ navigation completed in 47 steps

>>> Executing fetch_navigation...
...
✓ fetch_navigation completed in 52 steps

======================================================================
EXECUTION SUMMARY
======================================================================

navigation          | ✓ SUCCESS   | Steps:   47 | Distance:   0.30m
fetch_navigation    | ✓ SUCCESS   | Steps:   52 | Distance:   0.42m

======================================================================
Total Skills: 2
Successful: 2 (100.0%)
Total Steps: 99
======================================================================
```

## What Happens During Training

### 1. Model Checkpoints
Models are saved every 50k steps:
```
models/
├── lowlevel_curriculum_250k.zip           # Checkpoint
├── lowlevel_curriculum_checkpoints/
│   ├── lowlevel_curriculum_50k.zip
│   ├── lowlevel_curriculum_100k.zip
│   ├── lowlevel_curriculum_150k.zip
│   └── ...
└── fetch/
    ├── fetch_nav_250k_final.zip
    ├── fetch_nav_checkpoints/
    │   ├── fetch_nav_50k.zip
    │   └── ...
```

### 2. Training Logs
Detailed logs for each trainer:
```
logs/parallel/
├── nav_training_20240115_103045.log
└── fetch_training_20240115_103045.log

logs/fetch/
├── tensorboard/
│   └── events.out.tfevents.* 
└── fetch_nav_best/
    └── best_model.zip
```

### 3. TensorBoard Metrics (Docker)
Inside Docker container:
```bash
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```
Then open: http://localhost:6006

## After Training: Next Steps

### Option 1: Use Trained Models in High-Level HRL
```python
from src.skill_executor import SkillExecutor, NavigationSkill, FetchSkill

executor = SkillExecutor()
executor.add_skill(NavigationSkill("models/lowlevel_curriculum_250k"))
executor.add_skill(FetchSkill("models/fetch/fetch_nav_250k_final"))

# ... use in your high-level training
```

### Option 2: Evaluate Models
```bash
python src/evaluate_low_level_nav.py
python src/arm/evaluate_fetch.py
```

### Option 3: Continue Training from Checkpoint
```python
from stable_baselines3 import PPO
model = PPO.load("models/lowlevel_curriculum_250k")
model.learn(total_timesteps=100000)  # Train more steps
```

## Troubleshooting

### Issue: "Scene not found" error
**Cause:** Habitat dataset not downloaded
**Fix:** Run setup script to download datasets
```bash
bash setup_project.sh
```

### Issue: Out of memory (OOM) during training
**Cause:** Batch size too large for available memory
**Fix:** Modify training script
```python
# In train_*.py, change:
batch_size=32,  # reduced from 64
n_steps=1024,   # reduced from 2048
```

### Issue: Docker GPU not detected
**Cause:** nvidia-docker not installed
**Fix:**
```bash
# Check current setup
docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base nvidia-smi

# If error, install nvidia-container-runtime
# On Ubuntu: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
```

### Issue: Port 6006 already in use
**Cause:** Another service using TensorBoard port
**Fix:** Change port in docker-compose.yml
```yaml
ports:
  - "6007:6006"  # Use different host port
```

### Issue: Training stuck/hanging
**Cause:** Habitat simulator initialization slow on first run
**Fix:** This is normal - first reset takes longer. Be patient.
```
First reset: ~30 seconds (loading scene, physics, etc.)
Subsequent resets: ~1 second
```

## Performance Tips

### For CPU Training
- Close other applications to free memory
- Use smaller batch sizes if OOM
- Train overnight for longer experiments

### For GPU Training (Docker)
- Use NVIDIA GPU: `docker run --gpus all`
- Monitor GPU: `nvidia-smi` (inside container)
- Batch size can be larger: 128+ on V100

### For HPC (WPI Turing)
- Use GPU partition: `--partition=gpu --gpus=1`
- See `DOCKER.md` for SLURM job script
- Use `sbatch` for background training

## File Reference

| File | Purpose |
|------|---------|
| `launch_parallel_training.py` | Runs nav + fetch training simultaneously |
| `src/skill_executor.py` | Combines skills into sequences |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Multi-service configuration |
| `DOCKER.md` | Docker documentation |
| `IMPLEMENTATION_SUMMARY.md` | Detailed feature documentation |
| `.github/copilot-instructions.md` | AI agent guide |

## Development Workflow

```
1. Make code changes
   ↓
2. Test locally: python src/train_*.py
   ↓
3. Run parallel training: python launch_parallel_training.py
   ↓
4. Test skill integration: python src/skill_executor.py
   ↓
5. Integrate into high-level HRL
```

## Testing Your Setup

Run the integration test suite:
```bash
python test_integration.py
```

Expected output:
```
✓ SimpleNavigationEnv
✓ FetchNavigationEnv  
✓ SkillExecutor
✓ Dockerfile
✓ docker-compose.yml
...
✓✓✓ ALL TESTS PASSED ✓✓✓
```

## Common Commands Reference

```bash
# Train parallel
python launch_parallel_training.py --mode both

# Train individual
python src/train_low_level_nav.py
python src/arm/train_fetch_nav.py

# Test skill combo
python src/skill_executor.py

# Docker training
docker-compose up -d hrl-training
docker-compose exec hrl-training python launch_parallel_training.py

# View logs
tail -f logs/parallel/nav_training_*.log
docker-compose logs -f hrl-training

# Check models
ls -lh models/*.zip

# Clean up
docker-compose down
rm -rf logs/parallel/*
```

## Questions?

- **Architecture question?** → See `.github/copilot-instructions.md`
- **Docker issue?** → See `DOCKER.md`
- **Implementation details?** → See `IMPLEMENTATION_SUMMARY.md`
- **Environment error?** → Run `test_integration.py`

---

**Ready to start?** Pick an option above (A, B, or C) and begin! 🚀
