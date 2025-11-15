"""Place task implementation."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..task_base import TaskBase
from ..environment import Environment


class PlaceTask(TaskBase):
    """
    Task: Place a held object at a target location.

    Trained RL model learns to navigate to target and place object.
    """

    def __init__(
        self,
        env: Environment,
        model_path: Optional[str] = None,
        target_location: Optional[str] = None,
    ):
        """
        Initialize place task.

        Args:
            env: Environment instance
            model_path: Path to pre-trained model
            target_location: Name of target placement location
        """
        super().__init__(env=env, model_path=model_path, task_name="Place")
        self.target_location = target_location or "default"
        self.rnn_hidden_state = None
        self.max_steps = 150

    def load_model(self, model_path: str) -> None:
        """Load pre-trained place model."""
        try:
            self.model = torch.load(model_path, map_location="cpu")
            self.model.eval()
            print(f"Loaded place model from {model_path}")
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
        Get place action from policy.

        Moves arm to release position and opens gripper.
        """
        if self.model is not None:
            # Use learned policy
            obs_tensor = torch.from_numpy(
                observation.get("rgb", np.zeros((512, 512, 3)))
            ).float()
            # Placeholder policy inference
            action_array = np.random.randn(7)  # 7 DOF arm
        else:
            # Fallback heuristic: move arm to place position
            action_array = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        grip_action = -1.0  # Release

        return "arm_action", {
            "arm_action": action_array,
            "grip_action": grip_action,
        }

    def is_task_complete(self, info: Dict[str, Any]) -> bool:
        """Check if object successfully placed."""
        if "place_success" in info:
            return bool(info["place_success"])

        if self.step_count > self.max_steps:
            return True

        return False

    @property
    def goal_description(self) -> str:
        return f"Place at {self.target_location}"

