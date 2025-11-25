import gymnasium as gym
import numpy as np
import habitat_sim
import os
import quaternion  # numpy-quaternion


class SimpleNavigationEnv(gym.Env):
    """
    Habitat-Sim low-level navigation environment for PPO.

    Observation:
        [ distance_to_goal , relative_angle_to_goal ]

    Actions:
        0 = NO-OP (do nothing)
        1 = MOVE FORWARD
        2 = TURN LEFT
        3 = TURN RIGHT
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 scene_path="/Users/bluitel/Documents/WPI_DS551_G10_Final/habitat-lab/data/versioned_data/habitat_test_scenes/skokloster-castle.glb",
                 min_goal_dist=2.0,
                 max_goal_dist=8.0):

        super().__init__()

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene not found: {scene_path}")

        self.min_goal_dist = min_goal_dist
        self.max_goal_dist = max_goal_dist

        # ----------------------------------------------------
        # Habitat simulator setup
        # ----------------------------------------------------
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False
        backend_cfg.gpu_device_id = -1   # CPU rendering only

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # --- NO SENSORS AT ALL (OLD Habitat builds require this) ---
        agent_cfg.sensor_specifications = []

        # --- BASIC BODY PARAMETERS ---
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.1

        # --- MOVEMENT ACTIONS ---
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



        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        self.pathfinder = self.sim.pathfinder

        print("✓ Habitat simulator initialized.")

        # Gym spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi], dtype=np.float32),
            high=np.array([50.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_steps = 150     # with 0.5m forward, 150 steps is plenty
        self.current_step = 0
        self.goal_position = None
        self.prev_distance = None

    # --------------------------------------------------------
    # Helper: rotate vector by numpy-quaternion
    # --------------------------------------------------------
    def _quat_rotate(self, q_hab, v):
        """
        q_hab: habitat quaternion (with .w, .x, .y, .z)
        v: np.array([x,y,z])
        """
        q = np.quaternion(q_hab.w, q_hab.x, q_hab.y, q_hab.z)
        vq = np.quaternion(0.0, v[0], v[1], v[2])
        rq = q * vq * q.inverse()
        return np.array([rq.x, rq.y, rq.z], dtype=np.float32)

    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # Sample agent position
        agent_pos = self.pathfinder.get_random_navigable_point()

        # Curriculum-style goal sampling: keep goals within [min,max] distance
        for _ in range(50):
            goal = self.pathfinder.get_random_navigable_point()
            dist = np.linalg.norm(agent_pos - goal)
            if self.min_goal_dist <= dist <= self.max_goal_dist:
                self.goal_position = goal
                break
        else:
            # Fallback: just take whatever
            self.goal_position = goal

        # Set initial agent state (facing roughly -Z, but orientation doesn't matter
        # too much since we compute relative angle correctly)
        state = habitat_sim.AgentState()
        state.position = agent_pos
        state.rotation = habitat_sim.utils.quat_from_angle_axis(
            0.0, np.array([0.0, 1.0, 0.0])
        )
        self.agent.set_state(state)

        self.current_step = 0
        self.prev_distance = np.linalg.norm(agent_pos - self.goal_position)

        return self._get_obs(), {}

    # --------------------------------------------------------
    def step(self, action):
        # Map discrete action to Habitat actions
        if action == 1:
            self.agent.act("move_forward")
        elif action == 2:
            self.agent.act("turn_left")
        elif action == 3:
            self.agent.act("turn_right")
        # action 0 = NO-OP

        self.current_step += 1

        obs = self._get_obs()
        distance = float(obs[0])

        # ---------------- Reward shaping ----------------
        progress = self.prev_distance - distance   # >0 if we moved closer

        # Stronger shaping: reward progress and penalize distance and time
        reward = 5.0 * progress          # directional progress
        reward -= 0.001 * distance       # encourage staying close
        reward -= 0.01                   # time penalty

        done = False

        if distance < 0.5:
            reward += 10.0
            done = True

        truncated = self.current_step >= self.max_steps
        self.prev_distance = distance

        return obs, reward, done, truncated, {"distance": distance}

    # --------------------------------------------------------
    def _get_obs(self):
        state = self.agent.get_state()
        agent_pos = state.position
        quat = state.rotation

        # Distance to goal
        distance = np.linalg.norm(agent_pos - self.goal_position)

        # ---------- SAFE GUARDS ----------
        if distance < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        # ---------------------------------

        # Agent forward in world frame
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward = self._quat_rotate(quat, forward)
        forward[1] = 0.0

        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= np.linalg.norm(forward)

        # Direction to goal
        to_goal = self.goal_position - agent_pos
        to_goal[1] = 0.0

        if np.linalg.norm(to_goal) < 1e-6:
            to_goal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            to_goal /= np.linalg.norm(to_goal)

        # Signed angle
        cross = forward[0] * to_goal[2] - forward[2] * to_goal[0]
        dot = forward[0] * to_goal[0] + forward[2] * to_goal[2]
        angle = np.arctan2(cross, dot)

        # ---------- FINAL SAFETY ----------
        if np.isnan(angle):
            angle = 0.0
        if np.isnan(distance):
            distance = 50.0
        # ----------------------------------

        return np.array([distance, angle], dtype=np.float32)

    # --------------------------------------------------------
    def close(self):
        self.sim.close()
