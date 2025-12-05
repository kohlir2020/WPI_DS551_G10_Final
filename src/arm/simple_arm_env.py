import gymnasium as gym
import numpy as np
import habitat_sim
import magnum as mn
import os

class SimpleArmEnv(gym.Env):
    """
    Habitat-Sim Arm Manipulation Environment.
    
    Loads a Franka URDF directly and controls it to reach a virtual drawer.
    
    Observation:
        [EE_Position (3), Target_Position (3), Joint_Angles (7)]
        
    Action:
        [dx, dy, dz] - End Effector Velocity
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 urdf_path="data/robots/hab_franka/urdf/hab_franka.urdf",
                 max_steps=200):
        
        super().__init__()
        
        if not os.path.exists(urdf_path):
            # Fallback for standard habitat-lab paths if exact path isn't provided
            if os.path.exists("data/robots/hab_franka/urdf/hab_franka.urdf"):
                urdf_path = "data/robots/hab_franka/urdf/hab_franka.urdf"
            else:
                raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.max_steps = max_steps

        # ----------------------------------------------------
        # 1. Habitat Simulator Setup
        # ----------------------------------------------------
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = "NONE" # Empty scene
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [] # No sensors (blind RL)

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)

        # ----------------------------------------------------
        # 2. Load Robot via URDF
        # ----------------------------------------------------
        ao_mgr = self.sim.get_articulated_object_manager()
        self.robot = ao_mgr.add_articulated_object_from_urdf(
            filepath=self.urdf_path,
            fixed_base=True
        )
        self.robot.translation = np.array([0.0, 0.0, 0.0]) # Place at origin

        print(f"✓ Robot loaded from URDF. ID: {self.robot.object_id}")

        # Detect End-Effector Link (Last link in the chain)
        self.ee_link_id = self.robot.get_link_ids()[-1]
        
        # ----------------------------------------------------
        # 3. Gym Spaces
        # ----------------------------------------------------
        # Action: Velocity in X, Y, Z
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation: EE(3) + Target(3) + Joints(7)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Task Params
        self.drawer_pos = np.array([0.5, 0.4, -0.5], dtype=np.float32)
        self.current_step = 0
        self.home_joints = np.zeros(7, dtype=np.float32) # Simple home pose

    # --------------------------------------------------------
    # Helper: Numerical Jacobian (Robust IK)
    # --------------------------------------------------------
    def _compute_numerical_jacobian(self, joint_positions):
        """
        Calculates how much the EE moves for small changes in joints.
        Returns Matrix J (3x7).
        """
        n_joints = 7 # Franka has 7 arm joints
        J = np.zeros((3, n_joints), dtype=np.float32)
        epsilon = 1e-4

        # Save current state
        original_joints = self.robot.joint_positions
        
        # Get current EE pos
        self.robot.joint_positions = joint_positions
        p0 = self.robot.get_link_scene_node(self.ee_link_id).absolute_translation
        p0 = np.array(p0)

        # Perturb each joint
        for i in range(n_joints):
            # Create perturbed config
            temp_joints = np.array(joint_positions, copy=True)
            temp_joints[i] += epsilon
            
            # Apply and measure
            self.robot.joint_positions = temp_joints
            p_new = self.robot.get_link_scene_node(self.ee_link_id).absolute_translation
            p_new = np.array(p_new)
            
            # Finite difference
            J[:, i] = (p_new - p0) / epsilon

        # Restore state
        self.robot.joint_positions = original_joints
        return J

    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # Reset Robot Pose
        # Some randomness to prevent overfitting
        random_noise = np.random.uniform(-0.1, 0.1, 7)
        self.robot.joint_positions = self.home_joints + random_noise
        self.robot.joint_velocities = np.zeros(7)

        # Reset Drawer Position (Slight randomization)
        self.drawer_pos = np.array([0.5, 0.4, -0.5]) + np.random.uniform(-0.05, 0.05, 3)

        self.current_step = 0
        return self._get_obs(), {}

    # --------------------------------------------------------
    def step(self, action):
        # Action is EE Velocity [vx, vy, vz]
        target_vel = action * 0.1 # Slow down for stability
        
        # 1. Get Jacobian
        current_joints = np.array(self.robot.joint_positions)[:7] # First 7 are arm
        J = self._compute_numerical_jacobian(current_joints)
        
        # 2. Solve IK (Damped Least Squares) to get Joint Velocities
        # dq = J_T * inv(J*J_T + damp*I) * v
        damping = 0.05
        J_T = J.T
        inv_term = np.linalg.inv(J @ J_T + damping**2 * np.eye(3))
        dq = J_T @ inv_term @ target_vel
        
        # 3. Apply to Robot
        # We use Kinematic Control (setting positions directly) for simplicity
        # New Pose = Old Pose + Velocity
        new_joints = current_joints + dq
        
        # Clip limits (approximate for Franka)
        new_joints = np.clip(new_joints, -2.8, 2.8)
        
        # Create full joint array (padding if gripper exists, usually 2 extra joints)
        # We assume the first 7 are the arm.
        full_pose = np.zeros(len(self.robot.joint_positions))
        full_pose[:7] = new_joints
        self.robot.joint_positions = full_pose
        
        # 4. Step Simulation
        self.sim.step_physics(1.0 / 60.0)
        self.current_step += 1

        # 5. Check Success
        ee_pos = np.array(self.robot.get_link_scene_node(self.ee_link_id).absolute_translation)
        dist = np.linalg.norm(ee_pos - self.drawer_pos)

        reward = -dist # Minimize distance
        done = False
        
        # Success Logic
        if dist < 0.05: # 5cm threshold
            self._open_drawer()
            reward += 10.0
            done = True

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, truncated, {"distance": dist}

    # --------------------------------------------------------
    def _get_obs(self):
        ee_pos = np.array(self.robot.get_link_scene_node(self.ee_link_id).absolute_translation)
        joints = np.array(self.robot.joint_positions)[:7]
        
        return np.concatenate([
            ee_pos, 
            self.drawer_pos, 
            joints
        ]).astype(np.float32)

    def _open_drawer(self):
        """Logic to execute when we reach the drawer handle."""
        print(f"🎉 SUCCESS: Drawer opened at step {self.current_step}!")
        # If we had a drawer object, we would apply force here.

    def close(self):
        self.sim.close()