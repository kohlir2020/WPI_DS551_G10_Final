"""Habitat environment wrapper for multi-agent robot control."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import habitat
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from habitat.core.logging import logger

try:
    import pygame
except ImportError:
    pygame = None


class Environment:
    """
    Wrapper around habitat.Env that handles initialization,
    observation/action processing, and provides robot access.
    """

    def __init__(
        self,
        config_path: str = "benchmark/rearrange/play/play.yaml",
        camera_res: int = 512,
        with_render_camera: bool = True,
    ):
        """
        Initialize the Habitat environment.

        Args:
            config_path: Path to Habitat config YAML file
            camera_res: Resolution for render camera
            with_render_camera: Whether to add third-person camera for visualization
        """
        self.config_path = config_path
        self.camera_res = camera_res
        self.with_render_camera = with_render_camera
        self._env = None
        self._config = None
        self._init_environment()

    def _init_environment(self) -> None:
        """Initialize the Habitat environment with proper config."""
        self._config = habitat.get_config(self.config_path)

        with habitat.config.read_write(self._config):
            env_config = self._config.habitat.environment
            sim_config = self._config.habitat.simulator
            task_config = self._config.habitat.task

            if self.with_render_camera:
                sim_config.debug_render = True
                agent_config = get_agent_config(sim_config=sim_config)
                agent_config.sim_sensors.update(
                    {
                        "third_rgb_sensor": ThirdRGBSensorConfig(
                            height=self.camera_res, width=self.camera_res
                        )
                    }
                )

        self._env = habitat.Env(config=self._config)
        logger.info(f"Initialized Habitat environment from {self.config_path}")

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        obs = self._env.reset()
        return obs

    def step(
        self, action_name: str, action_args: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action_name: Name of the action (e.g., "arm_action", "base_velocity")
            action_args: Arguments for the action

        Returns:
            observations, reward, done, info
        """
        obs = self._env.step({"action": action_name, "action_args": action_args})
        reward = 0.0
        metrics = self._env.get_metrics()

        # Extract reward from metrics if available
        reward_keys = [k for k in metrics.keys() if "reward" in k.lower()]
        if reward_keys:
            reward = metrics[reward_keys[0]]

        done = self._env.episode_over
        info = metrics

        return obs, reward, done, info

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations without stepping."""
        return self._env.sim.get_sensor_observations()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current task metrics."""
        return self._env.get_metrics()

    @property
    def action_space(self):
        """Get environment action space."""
        return self._env.action_space

    @property
    def observation_space(self):
        """Get environment observation space."""
        return self._env.observation_space

    @property
    def sim(self):
        """Direct access to simulator."""
        return self._env._sim

    @property
    def task(self):
        """Direct access to task."""
        return self._env.task

    @property
    def articulated_agent(self):
        """Direct access to main articulated agent."""
        return self._env._sim.articulated_agent

    @property
    def is_multi_agent(self) -> bool:
        """Check if environment has multiple agents."""
        return len(self._env._sim.agents_mgr) > 1

    def render(self) -> Optional[np.ndarray]:
        """Render current observation as image."""
        from habitat.utils.visualizations.utils import observations_to_image

        obs = self.get_observations()
        metrics = self.get_metrics()
        image = observations_to_image(obs, metrics)
        return image

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

