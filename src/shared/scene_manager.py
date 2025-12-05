"""
Unified scene manager for consistent Fetch robot + Skokloster Castle setup
"""
import os
import habitat_sim
#from habitat.articulated_agents.robots import FetchRobot


def create_fetch_scene(scene_path=None, enable_physics=True, add_rgb_sensor=False):
    """
    Create Habitat simulator with Fetch robot in Skokloster Castle.
    
    Args:
        scene_path: Path to scene GLB (defaults to Skokloster Castle)
        enable_physics: Enable physics simulation
        add_rgb_sensor: Add RGB camera sensor for visualization
        
    Returns:
        (sim, agent, pathfinder): Simulator, agent, and pathfinder instances
    """
    if scene_path is None:
        scene_path = "habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    
    scene_path = os.path.abspath(scene_path)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    
    # Backend configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = enable_physics
    backend_cfg.gpu_device_id = -1  # CPU only
    
    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # RGB sensor (optional)
    if add_rgb_sensor:
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 1.25, 0.0]
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
    else:
        agent_cfg.sensor_specifications = []
    
    # Fetch robot actions (base movement)
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.5)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }
    
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.1
    
    # Create simulator
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(0)
    pathfinder = sim.pathfinder
    
    return sim, agent, pathfinder
