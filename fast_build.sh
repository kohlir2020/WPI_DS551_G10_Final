#!/bin/bash
# Fast Docker Build - Skips dataset download for quick startup
# Use this if you want to start training immediately

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║        ⚡ FAST DOCKER BUILD - Skips datasets for quick startup            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Stop existing
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build with fast Dockerfile
echo ""
echo "🔨 Building with Dockerfile.fast (faster, no dataset download)..."
echo "⏱️  Estimated time: 5-10 minutes"
echo ""

docker build -f Dockerfile.fast -t hrl-training:latest . --progress=plain

# Start container
echo ""
echo "🚀 Starting container..."
docker-compose up -d hrl-training

sleep 2

if docker ps -f name=hrl-training | grep -q hrl-training; then
    echo ""
    echo "✅ READY!"
    echo ""
    echo "Start training immediately:"
    echo "  docker exec hrl-training python src/arm/train_arm_multiagent.py --algorithm PPO --steps 100000 --device cuda"
    echo ""
else
    echo "✗ Container failed to start"
    docker logs hrl-training
fi
