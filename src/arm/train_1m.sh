#!/bin/bash
# 1M Training Script - Direct execution
source /opt/conda/etc/profile.d/conda.sh
conda activate hrl
cd /workspace

echo "======================================================================="
echo "1M STEP TRAINING - A2C + SAC"
echo "======================================================================="
echo "Start time: $(date)"
echo ""

# Train A2C for 1M steps
echo "PHASE 1: A2C Training - 1,000,000 steps"
python src/arm/train_habitat.py --algorithm A2C --steps 1000000 --device cuda
if [ $? -ne 0 ]; then
    echo "❌ A2C training failed!"
    exit 1
fi

echo ""
echo "✅ A2C completed!"
echo "⏳ Waiting 2 minutes before SAC..."
sleep 120

# Train SAC for 1M steps
echo ""
echo "PHASE 2: SAC Training - 1,000,000 steps"
python src/arm/train_habitat.py --algorithm SAC --steps 1000000 --device cuda
if [ $? -ne 0 ]; then
    echo "❌ SAC training failed!"
    exit 1
fi

echo ""
echo "======================================================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "======================================================================="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Compare results: python3 compare_all_phases.py --final"
echo "  2. Push to git: git add . && git commit -m '1M training' && git push"
