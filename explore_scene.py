#!/usr/bin/env python3
"""
Explore navigable areas in Skokloster Castle scene
"""
import habitat_sim
import numpy as np

# Load scene
scene_path = "habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
navmesh_path = "habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh"

# Create simulator
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = scene_path
agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

pathfinder = sim.pathfinder
if not pathfinder.is_loaded:
    print("⚠️  NavMesh not loaded, loading manually...")
    pathfinder.load_nav_mesh(navmesh_path)

print("="*70)
print("SKOKLOSTER CASTLE - NAVIGABLE AREA ANALYSIS")
print("="*70)

# Get bounds
bounds = pathfinder.get_bounds()
print(f"\nNavigable Area Bounds:")
print(f"  X: [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] ({bounds[1][0]-bounds[0][0]:.2f}m wide)")
print(f"  Y: [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] ({bounds[1][1]-bounds[0][1]:.2f}m tall)")
print(f"  Z: [{bounds[0][2]:.2f}, {bounds[1][2]:.2f}] ({bounds[1][2]-bounds[0][2]:.2f}m deep)")

# Sample navigable points
print(f"\nSample Navigable Points (20 random samples):")
samples = [pathfinder.get_random_navigable_point() for _ in range(20)]
for i, p in enumerate(samples):
    print(f"  {i+1:2d}. [{p[0]:6.2f}, {p[1]:5.2f}, {p[2]:6.2f}]")

# Cluster analysis
print(f"\nPoint Distribution Analysis:")
x_coords = [p[0] for p in samples]
y_coords = [p[1] for p in samples]
z_coords = [p[2] for p in samples]
print(f"  X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
print(f"  Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
print(f"  Z range: [{min(z_coords):.2f}, {max(z_coords):.2f}]")

# Test specific areas
print(f"\nTesting Common Areas:")
test_points = [
    ("Kitchen area", [8.0, 0.2, 5.0]),
    ("Dining room", [12.0, 0.2, 8.0]),
    ("Drawer area", [10.0, 0.2, 6.0]),
    ("Start position", [-16.0, 0.2, 5.0]),
    ("Center", [0.0, 0.2, 0.0]),
]

for name, point in test_points:
    snapped = pathfinder.snap_point(point)
    if not np.isnan(snapped).any():
        dist_to_orig = np.linalg.norm(np.array(point) - snapped)
        print(f"  ✓ {name:20s} navigable (snapped {dist_to_orig:.2f}m away)")
        print(f"    Original: {point}")
        print(f"    Snapped:  [{snapped[0]:.2f}, {snapped[1]:.2f}, {snapped[2]:.2f}]")
    else:
        print(f"  ✗ {name:20s} NOT navigable")

# Suggest good navigation targets
print(f"\n📍 SUGGESTED NAVIGATION TARGETS:")
print(f"  Based on actual scene analysis, use these coordinates:")
good_points = samples[:10]
for i, p in enumerate(good_points[:5]):
    print(f"  {i+1}. [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]")

sim.close()
print("\n✓ Analysis complete")
