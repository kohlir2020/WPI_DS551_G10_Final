
import habitat_sim
import numpy as np
import os

def inspect_fetch():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = -1
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    ao_mgr = sim.get_articulated_object_manager()
    urdf_path = "habitat-lab/data/robots/hab_fetch/robots/hab_fetch.urdf"
    urdf_path = os.path.abspath(urdf_path)
    
    robot = ao_mgr.add_articulated_object_from_urdf(urdf_path, fixed_base=True)
    
    link_ids = robot.get_link_ids()
    
    dof_index = 0
    arm_indices = []
    
    print("Mapping DOFs to Links:")
    for link_id in link_ids:
        # Check if link has a joint that adds a DOF
        # In habitat-sim, we can check joint type.
        # But python API might be limited.
        # However, we can check if the link is in the list of links that have DOFs.
        # Actually, get_link_ids returns all links.
        
        # Let's try to get the joint type.
        j_type = robot.get_link_joint_type(link_id)
        # JointType.Fixed is 0?
        
        # If it's not fixed, it consumes DOFs.
        # Most joints are 1 DOF (Revolute/Prismatic).
        
        # Let's print the type.
        name = robot.get_link_name(link_id)
        print(f"Link {link_id} ({name}): Type {j_type}")
        
    sim.close()

if __name__ == "__main__":
    inspect_fetch()
