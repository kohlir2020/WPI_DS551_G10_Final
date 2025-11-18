# envs/habitat_goal_nav_env.py
import gym
import numpy as np

class HabitatGoalNavEnv(gym.Env):
    """
    Goal-conditioned wrapper for Habitat navigation.
    Observation = dict(state, achieved_goal, desired_goal)
      - state: agent-centric representation (gps + compass)
      - achieved_goal: (x, y) current position
      - desired_goal: (x, y) target position
    Action = discrete {forward, turn_left, turn_right, stop}
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, habitat_env, success_dist=0.3):
        super().__init__()
        self.hab = habitat_env
        self.success_dist = success_dist

        # 4 primitive actions: forward, left, right, stop
        self.action_space = gym.spaces.Discrete(4)

        # State = [x, y, sin(theta), cos(theta)]
        self._state_dim = 4
        # Goal = (x, y)
        self._goal_dim = 2

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

        self._current_desired_goal = None

    def set_desired_goal(self, goal_xy: np.ndarray):
        """Set the final task goal (world or local frame)."""
        self._current_desired_goal = goal_xy.astype(np.float32)

    # ---- Helper encoders ----

    def _extract_agent_state(self, obs):
        gps = np.asarray(obs["gps"], dtype=np.float32)  # shape (2,)
        compass = float(obs["compass"][0])
        s = np.array(
            [gps[0], gps[1], np.sin(compass), np.cos(compass)],
            dtype=np.float32,
        )
        return s, gps  # state vector, position

    def _build_obs(self, raw_obs):
        state_vec, pos = self._extract_agent_state(raw_obs)
        achieved = pos.astype(np.float32)
        desired = self._current_desired_goal
        return {
            "observation": state_vec,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def compute_distance(self, ag, g):
        return float(np.linalg.norm(ag - g))

    # ---- Gym API ----

    def reset(self):
        raw_obs = self.hab.reset()

        # If no desired goal set externally, set one randomly for training
        if self._current_desired_goal is None:
            _, pos = self._extract_agent_state(raw_obs)
            # simple random target nearby (for vanilla training)
            offset = np.random.uniform(low=-1.5, high=1.5, size=(2,))
            self._current_desired_goal = (pos + offset).astype(np.float32)

        return self._build_obs(raw_obs)

    def step(self, action):
        if action == 0:
            act = {"action": "move_forward"}
        elif action == 1:
            act = {"action": "turn_left"}
        elif action == 2:
            act = {"action": "turn_right"}
        else:
            act = {"action": "stop"}

        raw_obs, _, done, info = self.hab.step(act)
        obs = self._build_obs(raw_obs)

        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        dist = self.compute_distance(ag, dg)

        success = dist < self.success_dist
        reward = 1.0 if success else -dist  # simple shaped reward

        info = dict(info)
        info["is_success"] = success
        info["distance_to_goal"] = dist

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.hab.render(mode=mode)
