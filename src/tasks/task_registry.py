"""Registry for task instantiation."""

from typing import Dict, Optional, Type

from ..task_base import TaskBase
from ..environment import Environment
from .pick_task import PickTask
from .place_task import PlaceTask
from .navigate_task import NavigateTask


class TaskRegistry:
    """Factory for creating task instances."""

    _registry: Dict[str, Type[TaskBase]] = {
        "pick": PickTask,
        "place": PlaceTask,
        "navigate": NavigateTask,
    }

    @classmethod
    def register(cls, name: str, task_class: Type[TaskBase]) -> None:
        """Register a new task class."""
        cls._registry[name.lower()] = task_class

    @classmethod
    def get_available_tasks(cls) -> list:
        """Get list of available task names."""
        return list(cls._registry.keys())

    @classmethod
    def create_task(
        cls,
        task_name: str,
        env: Environment,
        model_path: Optional[str] = None,
        **kwargs
    ) -> TaskBase:
        """
        Create a task instance.

        Args:
            task_name: Name of task (e.g., "pick", "place", "navigate")
            env: Environment instance
            model_path: Path to model checkpoint
            **kwargs: Additional task-specific arguments

        Returns:
            Task instance
        """
        task_name_lower = task_name.lower()
        if task_name_lower not in cls._registry:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {cls.get_available_tasks()}"
            )

        task_class = cls._registry[task_name_lower]
        return task_class(env=env, model_path=model_path, **kwargs)