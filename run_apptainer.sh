apptainer exec --nv ../hrl-training.sif bash -lc "
    source /opt/conda/etc/profile.d/conda.sh &&
    conda activate hrl &&
    python -u src/arm/hac_continuous_her_arm.py \
      --low_model_path=logs/simple_arm/cartesian_ppo_20251207_011850/final_ppo.zip
"
