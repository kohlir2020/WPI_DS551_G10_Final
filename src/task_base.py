"""Base class for RL-based tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from .environment import Environment


class TaskBase(ABC):
    """
    Abstract base class for RL-trained tasks.

    Each task wraps a pre-trained RL model and handles:
    - Model loading and initialization
    - Action execution in the environment
    - Observation/state handling
    - Reward/success tracking
    """

    def __init__(
        self,
        env: Environment,
        model_path: Optional[str] = None,
        task_name: str = "task",
    ):
        """
        Initialize task.

        Args:
            env: Environment instance to execute in
            model_path: Path to pre-trained model checkpoint
            task_name: Human-readable task name
        """
        self.env = env
        self.model_path = model_path
        self.task_name = task_name
        self.model = None
        self.is_done = False
        self.total_reward = 0.0
        self.step_count = 0

        if model_path:
            self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained RL model from checkpoint.

        Args:
            model_path: Path to model file
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get next action from RL policy given observation.

        Returns:
            (action_name, action_args) tuple for environment.step()
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset task-specific state (e.g., RNN hidden states)."""
        raise NotImplementedError

    @abstractmethod
    def is_task_complete(self, info: Dict[str, Any]) -> bool:
        """
        Check if task is successfully completed.

        Args:
            info: Metrics dictionary from environment

        Returns:
            True if task succeeded, False otherwise
        """
        raise NotImplementedError

    def step(
        self, observation: Dict[str, np.ndarray], task_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one step of the task.

        Args:
            observation: Current environment observation
            task_args: Optional task-specific arguments from VLM planner

        Returns:
            (observation, reward, done, info)
        """
        # Get action from policy
        action_name, action_args = self.get_action(observation)

        # Execute action in environment
        obs, reward, done, info = self.env.step(action_name, action_args)

        # Update tracking
        self.total_reward += reward
        self.step_count += 1

        # Check if task completed
        task_complete = self.is_task_complete(info) or done

        return obs, reward, task_complete, info

    @property
    def goal_description(self) -> str:
        """Return human-readable goal description."""
        return f"{self.task_name}: (implementation specific)"

    def get_state(self) -> Dict[str, Any]:
        """Get task state for serialization/debugging."""
        return {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "is_done": self.is_done,
        }

