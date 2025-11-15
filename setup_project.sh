if ! conda info --envs | grep -q "^habitat-lab-env[[:space:]]"; then
    conda create -n habitat-lab-env python=3.9 -c conda-forge
    conda env config vars set PYTHONNOUSERSITE=1 -n habitat-lab-env
    conda activate habitat-lab-env
    conda install habitat-sim withbullet -c conda-forge -c aihabitat
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    conda install -c conda-forge numpy quaternion numba llvmlite pygame pybullet wget -y
    cd habitat-lab 
    pip install -e habitat-lab
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/ 
    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data/
else
    echo "Conda environment 'habitat-lab-env' already exists. Skipping setup."
    conda activate habitat-lab-env
fi