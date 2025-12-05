#!/bin/bash
# Docker GPU Training Workflow
# Start container and train arm reaching with GPU support

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          GPU DOCKER TRAINING SETUP - ARM REACHING ENVIRONMENT             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

# ============================================================================
# STEP 1: Check GPU and Docker Setup
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Verifying GPU and Docker Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
    echo ""
else
    echo "✗ NVIDIA driver not found. Install nvidia-driver for GPU support."
    exit 1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA $CUDA_VERSION installed"
    echo ""
else
    echo "⚠ CUDA toolkit not found (optional for Docker)"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker installed: $(docker --version)"
else
    echo "✗ Docker not found. Install Docker to continue."
    exit 1
fi

# Check nvidia-docker
if docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base nvidia-smi &> /dev/null; then
    echo "✓ Docker GPU support enabled (nvidia-docker runtime)"
    echo ""
else
    echo "⚠ Docker GPU runtime not detected. May need to configure."
    echo "   See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    echo ""
fi

# ============================================================================
# STEP 2: Build Docker Image
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Building Docker Image (GPU-enabled)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$1" == "--rebuild" ]; then
    echo "Building with --no-cache (clean rebuild)..."
    docker-compose build --no-cache hrl-training
else
    echo "Building image (using cache if available)..."
    docker-compose build hrl-training
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Docker image built successfully"
else
    echo ""
    echo "✗ Docker build failed"
    exit 1
fi

# ============================================================================
# STEP 3: Start Container
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Starting Docker Container"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q '^hrl-training$'; then
    echo "Stopping existing container..."
    docker-compose down 2>/dev/null || true
    sleep 2
fi

# Start container
echo "Starting container..."
docker-compose up -d hrl-training

if docker ps --format '{{.Names}}' | grep -q '^hrl-training$'; then
    echo "✓ Container started successfully"
    echo ""
    echo "  Container ID: $(docker ps -q -f name=hrl-training)"
    echo "  Container name: hrl-training"
else
    echo "✗ Failed to start container"
    exit 1
fi

# ============================================================================
# STEP 4: Verify GPU in Container
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Verifying GPU Inside Container"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

docker exec hrl-training nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# ============================================================================
# STEP 5: Display Training Commands
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Ready to Train!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat << 'EOF'
🚀 Training Commands:

1. TRAIN WITH PPO (Recommended first)
   docker exec hrl-training python src/arm/train_arm_multiagent.py \
     --algorithm PPO \
     --steps 100000 \
     --device cuda

2. TRAIN WITH A2C (Actor-Critic)
   docker exec hrl-training python src/arm/train_arm_multiagent.py \
     --algorithm A2C \
     --steps 100000 \
     --device cuda

3. TRAIN WITH SAC (Soft Actor-Critic for continuous control)
   docker exec hrl-training python src/arm/train_arm_multiagent.py \
     --algorithm SAC \
     --steps 100000 \
     --device cuda

4. COMPARE ALL ALGORITHMS (Sequential training)
   docker exec hrl-training python src/arm/train_arm_multiagent.py \
     --compare \
     --compare-steps 50000 \
     --device cuda

📊 MONITORING:

Real-time TensorBoard (open in browser):
   tensorboard --logdir logs/fetch_arm

View logs in real-time:
   docker exec hrl-training tail -f logs/fetch_arm/*/events.out.tfevents*

Check GPU usage during training:
   watch -n 1 'docker exec hrl-training nvidia-smi'

📁 OUTPUT LOCATIONS:

- Models: logs/fetch_arm/{algorithm}_{timestamp}/
- TensorBoard logs: logs/fetch_arm/{algorithm}_{timestamp}/events.out.tfevents*
- Checkpoints: logs/fetch_arm/{algorithm}_{timestamp}/checkpoints/

🛑 STOP TRAINING:

Press Ctrl+C in the training terminal (graceful shutdown)
Or: docker kill hrl-training

📋 OTHER USEFUL COMMANDS:

Enter container shell:
   docker exec -it hrl-training bash

Interactive Python:
   docker exec -it hrl-training python

Run tests:
   docker exec hrl-training python src/arm/fetch_arm_reaching_env.py

View container logs:
   docker logs hrl-training

Stop container:
   docker-compose down

Remove container and image:
   docker-compose down --rmi all

EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next: Run a training command from above (e.g., PPO for first training)"
echo ""
