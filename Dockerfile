# Dockerfile for Hierarchical RL Training
# GPU-enabled with CUDA support for faster training

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    HABITAT_SIM_LOG=quiet \
    MAGNUM_LOG=quiet

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    ca-certificates \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libomp-dev \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    ffmpeg \
    xvfb \
    libgl1-mesa-glx libgl1-mesa-dev libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

# Create conda environment
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda create -n hrl python=3.9 -c conda-forge --override-channels -y
SHELL ["conda", "run", "-n", "hrl", "/bin/bash", "-c"]

# Install core dependencies
RUN pip install --upgrade pip && \
    pip install "numpy<2.0" quaternion numba llvmlite pygame pybullet

# Install Habitat packages
RUN conda install -c aihabitat -c conda-forge --override-channels\
    habitat-sim withbullet \
    -y

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install RL libraries
RUN pip install \
    "numpy<2.0" \
    stable-baselines3[extra] \
    gymnasium \
    tensorboard \
    wandb

# Install Habitat-Lab from source
RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git && \
    cd habitat-lab && \
    pip install -e . && \
    cd ..

# Create working directory
WORKDIR /workspace

# Download Habitat datasets
RUN python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /workspace/data/
RUN python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path /workspace/data/

# Create entrypoint script to keep container alive
RUN cat > /entrypoint.sh << 'EOFSCRIPT'
#!/bin/bash
echo "✓ Container ready for training"
echo "  Use: docker exec hrl-training python src/arm/train_arm_multiagent.py ..."
tail -f /dev/null
EOFSCRIPT
RUN chmod +x /entrypoint.sh

# Set entrypoint (keeps container alive while allowing docker exec)
ENTRYPOINT ["conda", "run", "-n", "hrl", "/entrypoint.sh"]
