from abc import ABC
import gym

class SkillTask(ABC):
    def __init__(self, algo_name: str, ckpt_path: str | None = None):
        ...

    # --- functions used for both ---
    def choose_action():
        pass

    def decode_obs():
        pass

    # --- training mode API ---
    def make_training_env(self) -> gym.Env:
        """
        This is a wrapper for the passed environment, specific to this task - 
        Not sure if the reward function should be defined in this wrapper or as its own function??
        """
        ...

    def train(self, total_steps: int):
        ...
    
    # --- execution mode API (HRL) ---
    def reset_runtime(self, env, goal):
        """Prepare to run on a *given* env with a specific goal."""
        ...

    def step_runtime(self, env) -> tuple[bool, dict]:
        """
        One step of this skill in the global env:
          - read obs from env
          - run policy
          - env.step(action)
          - return (skill_done, info)
        """
        ...
