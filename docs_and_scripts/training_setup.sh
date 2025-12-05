#!/bin/bash
# Complete Training Workflow for Hierarchical RL
# Trains Navigation + Arm Reaching skills in parallel

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# ============================================================================
# STEP 1: Verify Setup
# ============================================================================

print_header "STEP 1: Verifying Setup"

echo "Checking Python dependencies..."
if command -v python3 &> /dev/null; then
    print_success "Python 3 found: $(python3 --version)"
else
    print_error "Python 3 not found. Install Python 3.9+"
    exit 1
fi

echo "Checking project structure..."
required_files=(
    "src/simple_navigation_env.py"
    "src/train_low_level_nav.py"
    "src/arm/fetch_arm_reaching_env.py"
    "src/arm/train_fetch_arm.py"
    "src/skill_executor.py"
    "launch_parallel_training.py"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        print_success "$file"
    else
        print_error "Missing: $file"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    print_error "Some required files are missing"
    exit 1
fi

# ============================================================================
# STEP 2: Setup Directories
# ============================================================================

print_header "STEP 2: Setting Up Directories"

mkdir -p "$PROJECT_DIR/logs/parallel"
mkdir -p "$PROJECT_DIR/logs/fetch_arm"
mkdir -p "$PROJECT_DIR/models"

print_success "Created logs/parallel"
print_success "Created logs/fetch_arm"
print_success "Created models"

# ============================================================================
# STEP 3: Docker Check
# ============================================================================

print_header "STEP 3: Docker Configuration"

if command -v docker &> /dev/null; then
    print_success "Docker found: $(docker --version)"
    
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose found: $(docker-compose --version)"
        
        print_info "To run training in Docker:"
        echo "  docker-compose up -d"
        echo "  docker exec rl_training python src/arm/train_fetch_arm.py"
    fi
else
    print_error "Docker not found. To use GPU training, install Docker."
    echo ""
    echo "Alternatively, run locally with:"
    echo "  python src/arm/train_fetch_arm.py --steps 100000"
fi

# ============================================================================
# STEP 4: Validation
# ============================================================================

print_header "STEP 4: Code Validation"

echo "Running architecture validation..."
python "$PROJECT_DIR/validate_architecture.py"

# ============================================================================
# STEP 5: Training Options
# ============================================================================

print_header "STEP 5: Training Options"

echo "Choose training method:"
echo ""
echo "1. DOCKER TRAINING (Recommended - all deps included)"
echo "   docker-compose up -d"
echo "   docker exec rl_training python src/arm/train_fetch_arm.py --steps 100000"
echo ""
echo "2. LOCAL TRAINING (if dependencies installed)"
echo "   python src/arm/train_fetch_arm.py --steps 100000"
echo ""
echo "3. PARALLEL TRAINING (Navigation + Arm simultaneously)"
echo "   docker exec rl_training python launch_parallel_training.py --mode both"
echo ""
echo "4. CUSTOM TRAINING (modify parameters)"
echo "   python src/arm/train_fetch_arm.py \\"
echo "     --steps 200000 \\"
echo "     --lr 5e-4 \\"
echo "     --batch-size 64"
echo ""

# ============================================================================
# STEP 6: Monitoring
# ============================================================================

print_header "STEP 6: Monitoring Training"

echo "Commands to monitor training:"
echo ""
echo "Real-time TensorBoard:"
echo "  tensorboard --logdir logs/fetch_arm"
echo ""
echo "View latest logs:"
echo "  tail -30 logs/parallel/fetch_arm_training_*.log"
echo "  tail -30 logs/fetch_arm/*/events.out.tfevents*"
echo ""
echo "Check models saved:"
echo "  ls -lah logs/fetch_arm/*/checkpoints/"
echo ""
echo "Check training process:"
echo "  docker exec rl_training ps aux | grep train"
echo ""

# ============================================================================
# STEP 7: Integration
# ============================================================================

print_header "STEP 7: After Training"

echo "Once arm model is trained:"
echo ""
echo "1. View results:"
echo "   ls logs/fetch_arm/*/fetch_arm_reaching_final.zip"
echo ""
echo "2. Run combined skills:"
echo "   python src/skill_executor.py"
echo ""
echo "3. Test on specific task:"
echo "   python evaluate_combined_skills.py"
echo ""

# ============================================================================
# Summary
# ============================================================================

print_header "SETUP COMPLETE"

cat << 'EOF'
Architecture Summary:
  ✓ Navigation Environment: SimpleNavigationEnv (working)
  ✓ Arm Environment: FetchArmReachingEnv (ready)
  ✓ Navigation Trainer: train_low_level_nav.py (working)
  ✓ Arm Trainer: train_fetch_arm.py (ready)
  ✓ Skill Executor: skill_executor.py (ready)
  ✓ Parallel Launcher: launch_parallel_training.py (ready)

Key Features:
  ✓ Arm reaching with 7-DOF joints
  ✓ Automatic gripper trigger at 15cm
  ✓ PPO training with callbacks
  ✓ Checkpoint saving every 10k steps
  ✓ TensorBoard monitoring
  ✓ Parallel training support

Quick Start Commands:
  # In Docker (recommended)
  docker-compose up -d
  docker exec rl_training python src/arm/train_fetch_arm.py --steps 100000
  
  # Monitor training
  tensorboard --logdir logs/fetch_arm
  
  # Parallel training (nav + arm)
  docker exec rl_training python launch_parallel_training.py --mode both

Next Steps:
  1. Start Docker container
  2. Run arm training (100k steps minimum)
  3. Monitor with TensorBoard
  4. Integrate trained model into skill executor
  5. Test combined navigation + arm reaching + gripper trigger

Documentation:
  - QUICK_REFERENCE.md (this file)
  - ARM_REACHING_GUIDE.md (detailed guide)
  - ARCHITECTURE_REDESIGN.md (design explanation)
  - IMPLEMENTATION_COMPLETE.md (full summary)

Status: ✅ READY TO TRAIN
All files created, validated, and ready for execution.
EOF

echo ""
print_success "Setup complete! Ready to start training."
echo ""
