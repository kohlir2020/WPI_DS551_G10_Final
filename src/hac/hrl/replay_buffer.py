# hrl/replay_buffer.py
import numpy as np

class HindsightReplayBuffer:
    """
    Simple off-policy buffer with optional hindsight relabeling.
    Stores transitions (s, g, a, r, s', g', done, ag, ag')
    """

    def __init__(self, max_size, obs_dim, goal_dim, act_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.goal = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.int64)  # discrete indices
        self.rew = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_goal = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.achieved = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.next_achieved = np.zeros((max_size, goal_dim), dtype=np.float32)

    def add(self, obs, goal, act, rew, next_obs, next_goal, done, ag, next_ag):
        i = self.ptr
        self.obs[i] = obs
        self.goal[i] = goal
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.next_goal[i] = next_goal
        self.done[i] = done
        self.achieved[i] = ag
        self.next_achieved[i] = next_ag

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, her_ratio=0.8, success_func=None, reward_func=None):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = self.obs[idxs]
        goal = self.goal[idxs].copy()
        act = self.act[idxs]
        rew = self.rew[idxs].copy()
        next_obs = self.next_obs[idxs]
        next_goal = self.next_goal[idxs].copy()
        done = self.done[idxs]
        ag = self.achieved[idxs]
        next_ag = self.next_achieved[idxs]

        # HER relabeling: with probability her_ratio, use next_achieved as new goal
        if success_func is not None and reward_func is not None:
            her_mask = np.random.rand(batch_size) < her_ratio
            for i in range(batch_size):
                if her_mask[i]:
                    # new goal = next achieved
                    new_g = next_ag[i]
                    goal[i] = new_g
                    next_goal[i] = new_g
                    # recompute reward & done
                    success = success_func(next_ag[i], new_g)
                    rew[i] = reward_func(next_ag[i], new_g)
                    done[i] = float(success)

        return {
            "obs": obs,
            "goal": goal,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "next_goal": next_goal,
            "done": done,
        }
