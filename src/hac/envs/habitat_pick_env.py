# src/hac/envs/habitat_pick_env.py
import gym
import numpy as np

class HabitatPickEnv(gym.Env):
    """
    Goal-conditioned env for learning a PICK skill.
    Goal space: relative pose between gripper and target object (3D position).
    Success: gripper is grasping the object (info["is_grasped"] == True).
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, habitat_env, success_thresh=0.05):
        super().__init__()
        self.hab = habitat_env
        self.success_thresh = success_thresh

        # Action space: assume a discrete set of arm/gripper primitives you define
        # Example: 0: move_ee_forward, 1: back, 2: left, 3: right, 4: up, 5: down, 6: close_gripper
        self.action_space = gym.spaces.Discrete(7)

        # Observation:  gripper pos (3) + object pos (3) = 6-D
        self._state_dim = 6
        self._goal_dim = 3  # relative target offset

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
                ),
                "achieved_goal": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._goal_dim,), dtype=np.float32
                ),
                "desired_goal": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._goal_dim,), dtype=np.float32
                ),
            }
        )

        self._desired_goal = None

    # ---- helpers ----

    def _extract_positions(self, obs):
        """
        Adjust keys here to match your Habitat config.
        For many rearrange tasks, you'll have something like:
          obs["ee_pos"]       -> (3,) gripper position
          obs["object_pos"]   -> (3,) object position
        """
        ee = np.asarray(obs["ee_pos"], dtype=np.float32)
        obj = np.asarray(obs["object_pos"], dtype=np.float32)
        return ee, obj

    def _build_obs(self, raw_obs):
        ee, obj = self._extract_positions(raw_obs)
        state = np.concatenate([ee, obj], axis=0)
        # achieved_goal = relative offset object - gripper
        achieved = (obj - ee).astype(np.float32)
        desired = self._desired_goal
        return {
            "observation": state,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def compute_distance(self, ag, g):
        return float(np.linalg.norm(ag - g))

    # ---- gym API ----

    def reset(self):
        raw_obs = self.hab.reset()

        # If not set externally, default goal = zero offset (exact grasp)
        if self._desired_goal is None:
            ee, obj = self._extract_positions(raw_obs)
            self._desired_goal = (obj - ee).astype(np.float32)

        return self._build_obs(raw_obs)

    def set_desired_goal(self, goal_vec):
        """Goal is a relative offset we want between gripper and object."""
        self._desired_goal = goal_vec.astype(np.float32)

    def step(self, action):
        # Map discrete actions to Habitat actions:
        # You need to define these in your Habitat config (e.g. arm joint delta, gripper open/close).
        if action == 0:
            act = {"action": "ee_forward"}
        elif action == 1:
            act = {"action": "ee_back"}
        elif action == 2:
            act = {"action": "ee_left"}
        elif action == 3:
            act = {"action": "ee_right"}
        elif action == 4:
            act = {"action": "ee_up"}
        elif action == 5:
            act = {"action": "ee_down"}
        else:
            act = {"action": "grip_close"}

        raw_obs, _, done, info = self.hab.step(act)
        obs = self._build_obs(raw_obs)

        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        dist = self.compute_distance(ag, dg)

        # Success: either distance small, or Habitat says we're grasping
        grasp_flag = bool(info.get("is_grasped", False))
        success = dist < self.success_thresh or grasp_flag
        # Reward: shaped + bonus for success
        reward = -dist
        if success:
            reward += 1.0

        info = dict(info)
        info["is_success"] = success
        info["distance_to_goal"] = dist

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.hab.render(mode=mode)
