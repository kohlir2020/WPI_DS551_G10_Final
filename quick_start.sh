#!/bin/bash
# Quick Docker Start - Uses cached image if available
# Only rebuilds if absolutely necessary

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              QUICK DOCKER STARTUP - GPU TRAINING READY                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Step 1: Check if image exists
echo "📋 Checking for existing Docker image..."
if docker image inspect hrl-training:latest &> /dev/null; then
    echo "✓ Image hrl-training:latest found (using cached version)"
    BUILD_NEEDED=false
else
    echo "⚠ Image not found, will need to build"
    BUILD_NEEDED=true
fi

# Step 2: Stop any existing container
echo ""
echo "🛑 Stopping any existing containers..."
docker-compose down 2>/dev/null || true
sleep 1

# Step 3: Build if needed
if [ "$BUILD_NEEDED" = true ]; then
    echo ""
    echo "🔨 Building Docker image (this takes 5-10 minutes first time)..."
    echo "   Monitor with: docker ps"
    docker-compose build hrl-training
else
    echo "✓ Skipping build (using cached image)"
fi

# Step 4: Start container
echo ""
echo "🚀 Starting Docker container..."
docker-compose up -d hrl-training

# Step 5: Wait for container to be ready
echo ""
echo "⏳ Waiting for container to be ready..."
sleep 3

# Step 6: Verify container is running
if docker ps -f name=hrl-training | grep -q hrl-training; then
    echo "✓ Container started successfully!"
    CONTAINER_ID=$(docker ps -q -f name=hrl-training)
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "✅ READY FOR TRAINING!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Container ID: $CONTAINER_ID"
    echo "Container name: hrl-training"
    echo ""
    echo "🎯 Next: Run training command in another terminal:"
    echo ""
    echo "   docker exec hrl-training python src/arm/train_arm_multiagent.py \\"
    echo "     --algorithm PPO --steps 100000 --device cuda"
    echo ""
    echo "📊 Monitor with TensorBoard (in another terminal):"
    echo ""
    echo "   tensorboard --logdir logs/fetch_arm"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
else
    echo "✗ Container failed to start"
    echo ""
    echo "Debugging info:"
    docker ps -a -f name=hrl-training
    echo ""
    docker logs --tail 20 hrl-training
    exit 1
fi
