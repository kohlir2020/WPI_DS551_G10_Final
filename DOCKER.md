# Docker Setup for HRL Training

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start training container
docker-compose up -d hrl-training

# Enter the container
docker-compose exec hrl-training bash

# Inside container: run training
python launch_parallel_training.py --mode both
```

### Option 2: Direct Docker Commands

```bash
# Build image
docker build -t hrl-training:latest .

# Run training container with GPU
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  hrl-training:latest bash

# Inside container: run training
python launch_parallel_training.py --mode both
```

## Features

### GPU Support
- Built with CUDA 11.8 for GPU acceleration
- Automatically uses GPU if available
- Falls back to CPU if no GPU detected

### Volume Mounts
- `/workspace/src` - Your source code
- `/workspace/data` - Habitat datasets
- `/workspace/models` - Trained models
- `/workspace/logs` - Training logs
- `/workspace` - Full workspace access

### TensorBoard Monitoring
From container:
```bash
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

Then open: `http://localhost:6006` on your host machine

## Training Commands

### Inside Container

```bash
# Parallel training (Nav + Fetch)
python launch_parallel_training.py --mode both

# Navigation only
python src/train_low_level_nav.py

# Fetch navigation only
python src/arm/train_fetch_nav.py

# Evaluate trained models
python src/evaluate_low_level_nav.py
python src/arm/evaluate_fetch.py
```

### From Host (using docker-compose)

```bash
# Start training in background
docker-compose up -d hrl-training

# View logs in real-time
docker-compose logs -f hrl-training

# Stop training
docker-compose stop hrl-training

# Remove container
docker-compose down

# Execute command
docker-compose exec hrl-training python src/train_low_level_nav.py
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA Docker support
docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base nvidia-smi

# If this fails, install nvidia-docker:
# https://github.com/NVIDIA/nvidia-docker
```

### Out of Memory (OOM)
```python
# In training scripts, reduce batch size
batch_size = 32  # instead of 64
n_steps = 1024   # instead of 2048
```

### Slow Training on CPU
```bash
# Check CPU cores available
docker stats hrl-training

# For WPI Turing cluster with GPU:
srun --partition=gpu --gpus=1 docker-compose up -d hrl-training
```

## Advanced: Running on HPC (WPI Turing)

### Via SLURM Job Script

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hrl-training
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.log

# Load Docker/Singularity if needed
module load docker

# Start training
docker-compose up -d hrl-training
docker-compose exec hrl-training python launch_parallel_training.py --mode both

# Keep container running while job is active
while true; do
  docker-compose ps | grep -q hrl-training && sleep 60 || break
done
```

Run with:
```bash
sbatch run_hrl_training.sh
```

## File Structure Inside Container

```
/workspace/
├── src/
│   ├── simple_navigation_env.py
│   ├── hrl_highlevel_env.py
│   ├── train_low_level_nav.py
│   ├── train_high_level_improved.py
│   ├── arm/
│   │   ├── fetch_navigation_env.py
│   │   └── train_fetch_nav.py
│   └── tasks/
├── data/
│   ├── habitat_test_scenes/
│   └── replica_cad/
├── models/
│   ├── lowlevel_curriculum_250k.zip
│   ├── fetch/
│   │   └── fetch_nav_250k_final.zip
│   └── hl_improved/
├── logs/
│   ├── parallel/
│   ├── tensorboard/
│   └── fetch/
├── Dockerfile
├── docker-compose.yml
└── launch_parallel_training.py
```

## Environment Variables

Inside the container:
- `HABITAT_SIM_LOG=quiet` - Suppress Habitat logs
- `MAGNUM_LOG=quiet` - Suppress Magnum logs
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `CUDA_VISIBLE_DEVICES=0` - Specify GPU (if multiple)

Set custom variables:
```bash
docker-compose exec -e CUDA_VISIBLE_DEVICES=0 hrl-training bash
```

## Cleaning Up

```bash
# Stop all containers
docker-compose down

# Remove images
docker rmi hrl-training:latest

# Clean Docker build cache
docker system prune -a
```
