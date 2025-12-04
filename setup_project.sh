#!/bin/bash

# --- 1. Set Environment Variables for Conda Activation ---
# This is crucial for running 'conda activate' within a non-interactive script.
# Assumes miniconda/anaconda is in your PATH. If not, replace '/path/to/miniconda3'
# with the actual path to your Conda installation.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="habitat-lab-env"
PYTHON_VERSION="3.9"

echo "Checking for existing Conda environment '$ENV_NAME'..."

if ! conda info --envs | grep -q "^$ENV_NAME[[:space:]]"; then
    echo "Creating new Conda environment: $ENV_NAME with python=$PYTHON_VERSION"

    # --- 2. Create Environment and Activate ---
    # Use 'conda activate' after sourcing the init script above
    conda create -n $ENV_NAME python=$PYTHON_VERSION -c conda-forge -y
    conda activate $ENV_NAME

    # --- 3. Set Environment Variable inside the Conda Environment ---
    conda env config vars set PYTHONNOUSERSITE=1 -n $ENV_NAME

    echo "Installing Habitat-Sim and core dependencies..."

    # --- 4. Install Habitat-Sim and dependencies from multiple channels ---
    # Put 'aihabitat' and 'conda-forge' channels first for priority
    conda install \
        habitat-sim withbullet \
        numpy quaternion numba llvmlite pygame pybullet wget \
        -c aihabitat -c conda-forge -y

    # --- 5. Clone and Install Habitat-Lab ---
    echo "Cloning and installing habitat-lab..."
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    
    # Change into the cloned directory for pip install
    cd habitat-lab
    pip install -e . # Note: changed 'habitat-lab' to '.' which is common practice for -e install
    cd .. # Go back to original directory

    # --- 6. Download Datasets ---
    echo "Downloading Habitat datasets..."
    # Ensure this part is run while the environment is still active
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data/

    echo "Habitat Lab setup complete."
else
    echo "Conda environment '$ENV_NAME' already exists. Activating..."
    conda activate $ENV_NAME
fi

# Deactivate the environment after the script runs
# (or remove this if you want the user to stay in the env)
# conda deactivate