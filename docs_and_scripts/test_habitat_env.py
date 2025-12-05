import habitat_sim
import magnum as mn

def test():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # Force the GPU device 0 (Your GTX 1650)
    backend_cfg.gpu_device_id = 0
    
    # Create a dummy scene (no dataset required)
    backend_cfg.scene_id = "NONE"

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # Add a simple RGB sensor
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [640, 480]
    agent_cfg.sensor_specifications = [sensor_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

    try:
        print("Initializing Simulator...")
        sim = habitat_sim.Simulator(cfg)
        print("Simulator Initialized Successfully!")
        
        # Try one rendering step
        print("Attempting to render one frame...")
        obs = sim.get_sensor_observations()
        print("Frame rendered. Shape:", obs["color_sensor"].shape)
        
        sim.close()
        print("Test Passed.")
    except Exception as e:
        print("Test Failed with error:")
        print(e)

if __name__ == "__main__":
    test()