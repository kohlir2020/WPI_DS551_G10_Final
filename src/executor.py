"""Executor for running planned task sequences."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from habitat.core.logging import logger

from .environment import Environment
from .task_base import TaskBase
from .tasks import TaskRegistry


class Executor:
    """
    Orchestrates execution of a sequence of tasks.

    Handles state transitions between tasks, collects observations/rewards,
    and renders visual feedback.
    """

    def __init__(
        self,
        env: Environment,
        render: bool = True,
        save_video: bool = False,
        video_path: str = "./output.mp4",
    ):
        """
        Initialize executor.

        Args:
            env: Environment instance
            render: Whether to display rendering
            save_video: Whether to save video to file
            video_path: Path to save video
        """
        self.env = env
        self.render = render
        self.save_video = save_video
        self.video_path = video_path
        self.all_observations = []
        self.all_rewards = []
        self.task_history = []

    def execute(
        self, plan: List[Dict[str, Any]], max_steps_per_task: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a sequence of planned tasks.

        Args:
            plan: Task sequence from VLM planner
            max_steps_per_task: Maximum steps per task (for debugging)

        Returns:
            Execution statistics and results
        """
        logger.info(f"Starting execution of {len(plan)} tasks")

        # Reset environment
        obs = self.env.reset()
        execution_success = True
        total_reward = 0.0
        total_steps = 0

        for task_idx, task_plan in enumerate(plan):
            logger.info(f"\n--- Task {task_idx + 1}/{len(plan)} ---")
            logger.info(f"Task: {task_plan['task']}")
            logger.info(f"Description: {task_plan.get('description', 'N/A')}")
            logger.info(f"Args: {task_plan.get('args', {})}")

            # Create task instance
            task = TaskRegistry.create_task(
                task_plan["task"],
                self.env,
                task_args=task_plan.get("args", {}),
            )
            task.reset()

            # Execute task
            task_obs, task_reward, task_steps = self._run_task(
                task, obs, max_steps=max_steps_per_task
            )

            obs = task_obs
            total_reward += task_reward
            total_steps += task_steps
            execution_success = execution_success and (not task.is_done)

            # Record task result
            self.task_history.append(
                {
                    "task": task_plan["task"],
                    "reward": task_reward,
                    "steps": task_steps,
                    "success": not task.is_done,
                    "state": task.get_state(),
                }
            )

            logger.info(
                f"Task {task_idx + 1} complete. Reward: {task_reward:.3f}, Steps: {task_steps}"
            )

        # Finalize results
        results = {
            "total_reward": total_reward,
            "total_steps": total_steps,
            "plan_length": len(plan),
            "execution_success": execution_success,
            "task_history": self.task_history,
            "avg_reward_per_task": (
                total_reward / len(plan) if len(plan) > 0 else 0
            ),
        }

        if self.save_video:
            self._save_video()

        return results

    def _run_task(
        self,
        task: TaskBase,
        initial_obs: Dict[str, np.ndarray],
        max_steps: Optional[int] = None,
    ) -> tuple:
        """
        Run a single task to completion or max steps.

        Args:
            task: Task to execute
            initial_obs: Initial observation
            max_steps: Maximum steps for this task

        Returns:
            (final_observation, total_reward, steps_taken)
        """
        obs = initial_obs
        task_reward = 0.0
        step_count = 0
        max_steps = max_steps or 500

        while step_count < max_steps:
            # Get action from task policy
            obs, reward, done, info = task.step(obs)

            task_reward += reward
            step_count += 1

            # Render if enabled
            if self.render:
                self._render_step(obs, info, task.task_name, step_count)

            # Save observation if recording video
            if self.save_video:
                rendered = self.env.render()
                if rendered is not None:
                    self.all_observations.append(rendered)

            # Check task completion
            if done:
                logger.info(f"{task.task_name} completed in {step_count} steps")
                break

        self.all_rewards.append(task_reward)
        return obs, task_reward, step_count

    def _render_step(
        self, obs: Dict[str, np.ndarray], info: Dict[str, Any], task_name: str, step: int
    ) -> None:
        """Render current step (placeholder for visualization)."""
        try:
            from habitat.utils.visualizations.utils import (
                observations_to_image,
                overlay_frame,
            )

            image = observations_to_image(obs, info)
            
            # Add task name and step to image
            import cv2
            cv2.putText(
                image,
                f"{task_name} - Step {step}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            
            # TODO: Display image with pygame or save for video
            
        except ImportError:
            pass

    def _save_video(self) -> None:
        """Save collected observations as video."""
        if not self.all_observations:
            logger.warning("No observations to save")
            return

        try:
            from habitat_sim.utils import viz_utils as vut

            observations = np.array(self.all_observations)
            # observations shape: [num_frames, height, width, channels]
            observations = np.expand_dims(observations, 1)
            vut.make_video(
                observations, 0, "color", self.video_path
            )
            logger.info(f"Saved video to {self.video_path}")
        except Exception as e:
            logger.error(f"Could not save video: {e}")

    def get_results(self) -> Dict[str, Any]:
        """Get execution results and statistics."""
        return {
            "total_reward": sum(self.all_rewards),
            "num_tasks": len(self.task_history),
            "task_history": self.task_history,
            "avg_reward": (
                sum(self.all_rewards) / len(self.all_rewards)
                if self.all_rewards
                else 0
            ),
        }

