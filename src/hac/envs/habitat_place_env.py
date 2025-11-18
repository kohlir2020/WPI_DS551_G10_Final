# src/hac/envs/habitat_place_env.py
import gym
import numpy as np

class HabitatPlaceEnv(gym.Env):
    """
    Goal-conditioned env for PLACE skill.
    Goal space: relative offset between object and receptacle center.
    Success: object is within small distance of receptacle center.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, habitat_env, success_thresh=0.05):
        super().__init__()
        self.hab = habitat_env
        self.success_thresh = success_thresh

        # Action space reuses the same arm/gripper primitives as pick
        self.action_space = gym.spaces.Discrete(7)

        # Observation: object pos (3) + receptacle pos (3) + maybe ee pos (3) = 6 or 9
        self._state_dim = 6
        self._goal_dim = 3  # object relative to receptacle

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

    def _extract_positions(self, obs):
        """
        Adjust keys here to match your Habitat config.
        Example:
          obs["object_pos"]       -> (3,)
          obs["receptacle_pos"]   -> (3,)
        """
        obj = np.asarray(obs["object_pos"], dtype=np.float32)
        rec = np.asarray(obs["receptacle_pos"], dtype=np.float32)
        return obj, rec

    def _build_obs(self, raw_obs):
        obj, rec = self._extract_positions(raw_obs)
        state = np.concatenate([obj, rec], axis=0)
        achieved = (obj - rec).astype(np.float32)
        desired = self._desired_goal
        return {
            "observation": state,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def compute_distance(self, ag, g):
        return float(np.linalg.norm(ag - g))

    def reset(self):
        raw_obs = self.hab.reset()

        if self._desired_goal is None:
            obj, rec = self._extract_positions(raw_obs)
            self._desired_goal = (obj - rec).astype(np.float32)

        return self._build_obs(raw_obs)

    def set_desired_goal(self, goal_vec):
        self._desired_goal = goal_vec.astype(np.float32)

    def step(self, action):
        # Same primitive mapping as pick
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
            act = {"action": "grip_open"}  # for place, open the gripper

        raw_obs, _, done, info = self.hab.step(act)
        obs = self._build_obs(raw_obs)

        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        dist = self.compute_distance(ag, dg)

        # Success: object inside receptacle region or small distance
        place_flag = bool(info.get("is_placed", False))
        success = dist < self.success_thresh or place_flag
        reward = -dist
        if success:
            reward += 1.0

        info = dict(info)
        info["is_success"] = success
        info["distance_to_goal"] = dist

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.hab.render(mode=mode)
