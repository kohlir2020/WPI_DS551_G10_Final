#!/usr/bin/env python3
"""
Multi-task RL robot controller with VLM-based planning.

This program demonstrates:
1. Parse natural language robot command
2. VLM decomposes command into subtasks (pick, place, navigate)
3. Each trained RL model executes its subtask
4. Visual simulation shows task execution
5. Metrics and results displayed

Usage:
    python src/main.py --command "pick up the cup and place it on the table"
    python src/main.py --interactive
"""

import argparse
import os
from typing import Optional

from habitat.core.logging import logger

from environment import Environment
from executor import Executor
from vlm_planner import VLMPlanner
from tasks import TaskRegistry


def main(
    user_command: Optional[str] = None,
    config_path: str = "benchmark/rearrange/play/play.yaml",
    interactive: bool = False,
    render: bool = True,
    save_video: bool = False,
) -> None:
    """
    Main program flow.

    Args:
        user_command: Natural language command (if not interactive)
        config_path: Habitat environment config
        interactive: Whether to prompt user for command
        render: Whether to visualize execution
        save_video: Whether to save execution video
    """
    logger.info("=" * 60)
    logger.info("Multi-Task RL Robot Controller with VLM Planning")
    logger.info("=" * 60)

    # Initialize environment
    logger.info("\n[1] Initializing Habitat Environment...")
    with Environment(config_path=config_path) as env:
        # Get available tasks
        available_tasks = TaskRegistry.get_available_tasks()
        logger.info(f"Available tasks: {available_tasks}")

        # Initialize VLM planner
        logger.info("\n[2] Initializing VLM Planner...")
        try:
            vlm_planner = VLMPlanner(
                available_tasks=available_tasks,
                model="gpt-3.5-turbo",  # Use cheaper model for faster response
            )
            logger.info("VLM Planner ready (using OpenAI API)")
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            logger.info("Falling back to heuristic planning")
            vlm_planner = VLMPlanner(
                available_tasks=available_tasks,
                model="fallback",
            )

        # Get user command
        logger.info("\n[3] Getting Robot Command...")
        if interactive:
            user_command = input("Enter robot command: ").strip()
        elif user_command is None:
            user_command = "pick up the cube"
            logger.info(f"Using default command: {user_command}")

        logger.info(f"User command: '{user_command}'")

        # Plan task sequence using VLM
        logger.info("\n[4] Planning Task Sequence with VLM...")
        plan = vlm_planner.plan(user_command)
        logger.info(f"Generated plan with {len(plan)} tasks:")
        for i, task in enumerate(plan):
            logger.info(f"  {i+1}. {task['task']}: {task.get('description', '')}")

        # Execute planned tasks
        logger.info("\n[5] Executing Task Plan...")
        executor = Executor(env=env, render=render, save_video=save_video)
        results = executor.execute(plan)

        # Display results
        logger.info("\n[6] Execution Complete!")
        logger.info("=" * 60)
        logger.info("Results:")
        logger.info(f"  Total Reward: {results['total_reward']:.3f}")
        logger.info(f"  Total Steps: {results['total_steps']}")
        logger.info(f"  Plan Length: {results['plan_length']}")
        logger.info(
            f"  Execution Success: {'Yes' if results['execution_success'] else 'No'}"
        )
        logger.info(f"  Avg Reward per Task: {results['avg_reward_per_task']:.3f}")
        logger.info("\nTask Breakdown:")
        for task in results["task_history"]:
            logger.info(
                f"  {task['task']}: reward={task['reward']:.3f}, "
                f"steps={task['steps']}, success={task['success']}"
            )
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-task RL robot controller with VLM planning"
    )
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Natural language robot command (e.g., 'pick up the cup')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt user for command instead of using default",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark/rearrange/play/play.yaml",
        help="Habitat environment config file",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visual rendering",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save execution video to file",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="./robot_execution.mp4",
        help="Path to save video",
    )

    args = parser.parse_args()

    main(
        user_command=args.command,
        config_path=args.config,
        interactive=args.interactive,
        render=not args.no_render,
        save_video=args.save_video,
    )