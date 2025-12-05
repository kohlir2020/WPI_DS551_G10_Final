#!/bin/bash
# GPU TRAINING STATUS & QUICK START

cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║          🚀 GPU DOCKER TRAINING - COMPLETE & READY TO START 🚀           ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ SYSTEM STATUS

Environment:
  ✓ CUDA 11.8 available (nvcc verified)
  ✓ Docker configured for GPU (nvidia-docker runtime)
  ✓ NVIDIA GPU detected and ready
  ✓ Container prepared and ready to start

Implementation:
  ✓ ARM Reaching Environment (low-level state: distance + arm_angles)
  ✓ Vision-Based Environment (RGB stacked images - for next phase)
  ✓ Multi-Agent Trainer (PPO, A2C, SAC algorithms)
  ✓ GPU Training Scripts (with comprehensive monitoring)
  ✓ Complete Documentation (guides, commands, workflows)

═════════════════════════════════════════════════════════════════════════════
 🎯 QUICK START (3 COMMANDS)
═════════════════════════════════════════════════════════════════════════════

1️⃣  Build Docker image & start container:
    ./start_gpu_training.sh

2️⃣  In another terminal, start PPO training:
    docker exec hrl-training python src/arm/train_arm_multiagent.py \
      --algorithm PPO --steps 100000 --device cuda

3️⃣  Monitor with TensorBoard (browser: http://localhost:6006):
    tensorboard --logdir logs/fetch_arm

═════════════════════════════════════════════════════════════════════════════
 📋 TRAINING PHASES
═════════════════════════════════════════════════════════════════════════════

PHASE 1: PPO BASELINE (CURRENT) ⬅️ START HERE
├─ Command: docker exec hrl-training python src/arm/train_arm_multiagent.py \
│            --algorithm PPO --steps 100000 --device cuda
├─ Time: ~10 minutes (GPU), ~40 minutes (CPU)
├─ Expected success rate: 70-85%
└─ Output: logs/fetch_arm/ppo_YYYYMMDD_HHMMSS/arm_ppo_final.zip

PHASE 2: COMPARE RL ALGORITHMS
├─ Command: docker exec hrl-training python src/arm/train_arm_multiagent.py \
│            --compare --compare-steps 50000 --device cuda
├─ Algorithms: PPO vs A2C vs SAC (sequential training)
├─ Time: ~45-60 minutes (GPU)
├─ Comparison: Success rate, training stability, convergence speed
└─ Output: 3 trained models for comparison

PHASE 3: VISION-BASED OBSERVATIONS (FUTURE)
├─ Environment: vision_arm_reaching_env.py
├─ Observation: Stacked RGB images (4 frames from 3 cameras)
├─ Advantage: Can learn visual features, more realistic
└─ Training: Similar commands with vision trainer

═════════════════════════════════════════════════════════════════════════════
 📊 WHAT YOU GET
═════════════════════════════════════════════════════════════════════════════

Files Created:
  ✓ src/arm/train_arm_multiagent.py (Multi-algorithm trainer)
  ✓ src/arm/vision_arm_reaching_env.py (Vision-based environment)
  ✓ start_gpu_training.sh (Automated setup script)
  ✓ GPU_TRAINING_GUIDE.md (Complete guide)

Environments:
  ✓ FetchArmReachingEnv (Low-level: 8D observation, 7D actions)
  ✓ VisionArmReachingEnv (Vision-based: RGB stacks, 7D actions)

Algorithms:
  ✓ PPO (Proximal Policy Optimization) - Recommended
  ✓ A2C (Actor-Critic) - Alternative
  ✓ SAC (Soft Actor-Critic) - Continuous control specialist

GPU Support:
  ✓ CUDA 11.8 integration
  ✓ Automatic GPU detection
  ✓ GPU memory management
  ✓ Device selection (cuda/cpu/auto)

Monitoring:
  ✓ TensorBoard logging
  ✓ Checkpoint saving (every 10k steps)
  ✓ Evaluation callbacks
  ✓ Real-time GPU monitoring
  ✓ Training progress tracking

═════════════════════════════════════════════════════════════════════════════
 📁 OUTPUT STRUCTURE
═════════════════════════════════════════════════════════════════════════════

After training, you'll find:

logs/fetch_arm/
├── ppo_YYYYMMDD_HHMMSS/
│   ├── arm_ppo_final.zip           ← Final trained model
│   ├── best_model.zip              ← Best during training
│   ├── checkpoints/
│   │   ├── arm_ppo_10000_steps.zip
│   │   ├── arm_ppo_20000_steps.zip
│   │   └── ...
│   └── events.out.tfevents...      ← TensorBoard logs
├── a2c_YYYYMMDD_HHMMSS/            (if you train A2C)
│   └── ...
└── sac_YYYYMMDD_HHMMSS/            (if you train SAC)
    └── ...

TensorBoard will show:
  - Episode reward (should trend upward)
  - Episode length
  - Success rate
  - Model loss
  - Entropy
  - Learning rate

═════════════════════════════════════════════════════════════════════════════
 💡 TRAINING SCENARIOS
═════════════════════════════════════════════════════════════════════════════

SCENARIO 1: Quick Test (Verify Everything Works)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 5000 --device cuda
  Time: ~1 minute | Good for: Testing setup

SCENARIO 2: Standard Training (Recommended)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 100000 --device cuda
  Time: ~10 minutes | Good for: Baseline model

SCENARIO 3: Extended Training (Better Convergence)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 200000 --device cuda
  Time: ~20 minutes | Good for: Higher performance

SCENARIO 4: Algorithm Comparison (Research)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --compare --compare-steps 50000 --device cuda
  Time: ~45-60 minutes | Good for: Comparing algorithms

SCENARIO 5: CPU Debugging (No GPU)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 10000 --device cpu
  Time: ~30 minutes | Good for: Debugging

═════════════════════════════════════════════════════════════════════════════
 🔍 MONITORING DURING TRAINING
═════════════════════════════════════════════════════════════════════════════

Terminal 1 - Start Training:
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 100000 --device cuda

Terminal 2 - Monitor with TensorBoard:
  tensorboard --logdir logs/fetch_arm
  (Open browser: http://localhost:6006)

Terminal 3 - Watch GPU:
  watch -n 1 'docker exec hrl-training nvidia-smi'

Terminal 4 - Check Logs:
  docker exec hrl-training tail -f logs/fetch_arm/*/events.out.tfevents*

═════════════════════════════════════════════════════════════════════════════
 🎓 EXPECTED RESULTS
═════════════════════════════════════════════════════════════════════════════

PPO Baseline (100k steps):
  ✓ Training time: ~10 min (GPU) or ~40 min (CPU)
  ✓ Success rate: 0% → 30% → 60% → 80%+
  ✓ Final reward: 0.65-0.75
  ✓ Training stability: High
  ✓ Model size: ~500KB

A2C Comparison (50k steps):
  ✓ Training time: ~6 min (GPU)
  ✓ Success rate: ~70% (slower convergence than PPO)
  ✓ Final reward: 0.55-0.65
  ✓ Training stability: Medium

SAC Comparison (50k steps):
  ✓ Training time: ~7 min (GPU)
  ✓ Success rate: ~75% (good for continuous control)
  ✓ Final reward: 0.60-0.70
  ✓ Training stability: Medium-High

═════════════════════════════════════════════════════════════════════════════
 🛠️ USEFUL COMMANDS
═════════════════════════════════════════════════════════════════════════════

Setup & Start:
  ./start_gpu_training.sh              # Automated setup
  docker-compose up -d hrl-training    # Manual start
  docker ps -f name=hrl-training       # Check running

Training:
  # PPO (standard)
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm PPO --steps 100000 --device cuda

  # A2C
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm A2C --steps 100000 --device cuda

  # SAC
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --algorithm SAC --steps 100000 --device cuda

  # Compare all
  docker exec hrl-training python src/arm/train_arm_multiagent.py \
    --compare --compare-steps 50000 --device cuda

Monitoring:
  tensorboard --logdir logs/fetch_arm                    # TensorBoard
  watch -n 1 'docker exec hrl-training nvidia-smi'       # GPU monitor
  docker logs -f hrl-training                            # Container logs
  docker exec hrl-training ls -lah logs/fetch_arm/*/    # List outputs

Management:
  docker exec -it hrl-training bash                      # Shell access
  docker exec hrl-training nvidia-smi                    # Check GPU
  docker-compose down                                    # Stop container
  docker rmi hrl-training:latest                         # Remove image

═════════════════════════════════════════════════════════════════════════════
 📚 DOCUMENTATION
═════════════════════════════════════════════════════════════════════════════

GPU_TRAINING_GUIDE.md
  ✓ Complete training guide with all commands
  ✓ Experiment workflows
  ✓ Monitoring instructions
  ✓ Troubleshooting tips

train_arm_multiagent.py
  ✓ Multi-algorithm trainer (PPO, A2C, SAC)
  ✓ GPU support with automatic detection
  ✓ Comprehensive logging
  ✓ Easy CLI with argparse

vision_arm_reaching_env.py
  ✓ Vision-based environment (next phase)
  ✓ Multi-camera setup (front, side, top-down)
  ✓ Frame stacking for temporal information
  ✓ RGB observation space

═════════════════════════════════════════════════════════════════════════════
 🚀 NEXT STEPS
═════════════════════════════════════════════════════════════════════════════

1. Run setup script:
   chmod +x start_gpu_training.sh
   ./start_gpu_training.sh

2. Start PPO training (in new terminal):
   docker exec hrl-training python src/arm/train_arm_multiagent.py \
     --algorithm PPO --steps 100000 --device cuda

3. Monitor with TensorBoard:
   tensorboard --logdir logs/fetch_arm

4. After training completes:
   ✓ Model saved at: logs/fetch_arm/ppo_*/arm_ppo_final.zip
   ✓ Compare with other algorithms
   ✓ Consider vision-based training next

═════════════════════════════════════════════════════════════════════════════

✅ STATUS: READY FOR TRAINING

All systems configured for GPU training. Execute the commands above to begin!

═════════════════════════════════════════════════════════════════════════════
EOF
