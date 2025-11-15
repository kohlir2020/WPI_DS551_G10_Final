"""Navigation task implementation."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..task_base import TaskBase
from ..environment import Environment


class NavigateTask(TaskBase):
    """
    Task: Navigate to a target location.

    Trained RL model learns base velocity control for navigation.
    """

    def __init__(
        self,
        env: Environment,
        model_path: Optional[str] = None,
        target_location: Optional[str] = None,
    ):
        """
        Initialize navigate task.

        Args:
            env: Environment instance
            model_path: Path to pre-trained model
            target_location: Name of target location
        """
        super().__init__(env=env, model_path=model_path, task_name="Navigate")
        self.target_location = target_location or "default"
        self.rnn_hidden_state = None
        self.max_steps = 200

    def load_model(self, model_path: str) -> None:
        """Load pre-trained navigation model."""
        try:
            self.model = torch.load(model_path, map_location="cpu")
            self.model.eval()
            print(f"Loaded navigate model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.model = None

    def reset(self) -> None:
        """Reset task state."""
        self.is_done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.rnn_hidden_state = None

    def get_action(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get navigation action from policy.

        Controls base velocity [forward, turn].
        """
        if self.model is not None:
            # Use learned policy
            obs_tensor = torch.from_numpy(
                observation.get("rgb", np.zeros((512, 512, 3)))
            ).float()
            # Placeholder policy inference
            action_array = np.random.randn(2)  # [forward, turn]
        else:
            # Fallback heuristic: move forward
            action_array = np.array([0.5, 0.0])

        return "base_velocity", {"base_vel": action_array}

    def is_task_complete(self, info: Dict[str, Any]) -> bool:
        """Check if navigation target reached."""
        if "nav_success" in info:
            return bool(info["nav_success"])

        if self.step_count > self.max_steps:
            return True

        return False

    @property
    def goal_description(self) -> str:
        return f"Navigate to {self.target_location}"

