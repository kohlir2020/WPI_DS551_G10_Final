# test_habitat_setup.py
"""
Quick diagnostic script to see what's working in Habitat setup
"""

print("=" * 60)
print("HABITAT INSTALLATION CHECK")
print("=" * 60)

# 1. Check Python packages
print("\n[1] Checking installed packages...")
try:
    import habitat
    print(f"habitat-lab version: {habitat.__version__}")
except ImportError as e:
    print(f"habitat-lab not found: {e}")

try:
    import habitat_sim
    print(f"habitat-sim version: {habitat_sim.__version__}")
except ImportError as e:
    print(f"habitat-sim not found: {e}")

# 2. Check CUDA
print("\n[2] Checking CUDA setup...")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU (slower but will work)")

# 3. Check available scenes
print("\n[3] Checking available scenes...")
import os
import habitat
from pathlib import Path

# Try to find data directory
possible_data_dirs = [
    "data",
    os.path.expanduser("~/habitat-lab/data"),
    os.path.expanduser("~/.habitat/data"),
]

for data_dir in possible_data_dirs:
    if os.path.exists(data_dir):
        print(f"✓ Found data directory: {data_dir}")
        
        # Look for scene files
        scene_dir = os.path.join(data_dir, "scene_datasets")
        if os.path.exists(scene_dir):
            print(f"  Scene datasets:")
            for item in os.listdir(scene_dir):
                print(f"    - {item}")
        break
else:
    print("✗ No data directory found")

# 4. Try to load a simple environment (CPU mode)
print("\n[4] Testing basic environment loading (CPU mode)...")
try:
    from habitat.config.default import get_config
    from habitat import Env
    
    # Use CPU-only config
    config = get_config()
    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = -1  # Force CPU
    config.TASK.TYPE = "Nav-v0"
    config.SIMULATOR.SCENE = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    config.freeze()
    
    env = Env(config=config)
    obs = env.reset()
    
    print(f"Environment loaded successfully!")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Action space: {env.action_space}")
    
    env.close()
    
except Exception as e:
    print(f"Failed to load environment: {e}")
    print(f"  Error details: {type(e).__name__}")

# 5. Check if we can run the Fetch robot
print("\n[5] Checking Fetch robot availability...")
try:
    import habitat.articulated_agents
    print(" Articulated agents module available")
    
    # Check for robot configs
    robot_config_dir = Path(habitat.__file__).parent / "config" / "agents"
    if robot_config_dir.exists():
        print(f"  Robot configs available:")
        for config_file in robot_config_dir.glob("*.yaml"):
            print(f"    - {config_file.stem}")
    
except Exception as e:
    print(f"✗ Issue with articulated agents: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)