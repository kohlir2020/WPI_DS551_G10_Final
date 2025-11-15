"""Vision-Language Model planner for task decomposition."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from habitat.core.logging import logger

try:
    import openai
except ImportError:
    openai = None


class VLMPlanner:
    """
    Uses OpenAI GPT to decompose natural language commands into subtasks.

    Given a user's natural language instruction and available tasks,
    returns an ordered sequence of subtasks with parameters.
    """

    def __init__(
        self,
        available_tasks: List[str],
        api_key: Optional[str] = None,
        model: str = "gpt-4",
    ):
        """
        Initialize VLM planner.

        Args:
            available_tasks: List of available task names (e.g., ["pick", "place", "navigate"])
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: OpenAI model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        if openai is None:
            raise ImportError("openai package required. Install with: pip install openai")

        self.available_tasks = available_tasks
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def plan(self, user_command: str) -> List[Dict[str, Any]]:
        """
        Decompose natural language command into task sequence.

        Args:
            user_command: Natural language instruction from user

        Returns:
            List of task dictionaries with structure:
            [
                {
                    "task": "task_name",
                    "args": {"param1": value1, "param2": value2},
                    "description": "human readable description"
                },
                ...
            ]
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_command)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content
            logger.info(f"VLM Response:\n{response_text}")

            tasks = self._parse_response(response_text)
            return tasks

        except Exception as e:
            logger.error(f"VLM planning error: {e}")
            # Fallback to simple heuristic
            return self._fallback_plan(user_command)

    def _build_system_prompt(self) -> str:
        """Build system prompt for task decomposition."""
        tasks_list = ", ".join(self.available_tasks)
        return f"""You are a robot task planner. Your job is to decompose natural language 
commands into a sequence of executable subtasks.

Available tasks: {tasks_list}

For each task, you can specify parameters:
- pick: target_object (string)
- place: target_location (string)
- navigate: target_location (string)

Output format: Return a JSON array of task objects. Each object has:
- "task": the task name
- "args": a dict of parameters
- "description": brief explanation

Example output for "pick up the cup and place it on the table":
[
  {{"task": "pick", "args": {{"target_object": "cup"}}, "description": "Pick up the cup"}},
  {{"task": "place", "args": {{"target_location": "table"}}, "description": "Place cup on table"}}
]

IMPORTANT: Return ONLY valid JSON, no other text."""

    def _build_user_prompt(self, user_command: str) -> str:
        """Build user prompt for a specific command."""
        return f"""Decompose this robot command into subtasks:
"{user_command}"

Return only the JSON array of tasks."""

    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse VLM response into task list."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                tasks = json.loads(json_str)

                # Validate task structure
                validated_tasks = []
                for task in tasks:
                    if isinstance(task, dict) and "task" in task and "args" in task:
                        validated_tasks.append(
                            {
                                "task": task["task"],
                                "args": task.get("args", {}),
                                "description": task.get("description", ""),
                            }
                        )

                if validated_tasks:
                    return validated_tasks

            raise ValueError("Could not parse valid tasks from response")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Parse error: {e}. Using fallback.")
            return self._fallback_plan(response_text)

    def _fallback_plan(self, user_command: str) -> List[Dict[str, Any]]:
        """
        Fallback heuristic-based planning if VLM fails.

        Simple keyword matching to extract tasks.
        """
        command_lower = user_command.lower()
        tasks = []

        # Simple keyword matching
        if "pick" in command_lower or "grab" in command_lower:
            obj = self._extract_object(command_lower)
            tasks.append(
                {
                    "task": "pick",
                    "args": {"target_object": obj},
                    "description": f"Pick up the {obj}",
                }
            )

        if "place" in command_lower or "put" in command_lower:
            loc = self._extract_location(command_lower)
            tasks.append(
                {
                    "task": "place",
                    "args": {"target_location": loc},
                    "description": f"Place at {loc}",
                }
            )

        if "navigate" in command_lower or "go to" in command_lower or "move to" in command_lower:
            loc = self._extract_location(command_lower)
            tasks.append(
                {
                    "task": "navigate",
                    "args": {"target_location": loc},
                    "description": f"Navigate to {loc}",
                }
            )

        # Default to pick if nothing matched
        if not tasks:
            tasks.append(
                {
                    "task": "pick",
                    "args": {"target_object": "object"},
                    "description": "Pick up object",
                }
            )

        return tasks

    @staticmethod
    def _extract_object(text: str) -> str:
        """Simple heuristic to extract object name."""
        keywords = ["cup", "bottle", "box", "ball", "object"]
        for keyword in keywords:
            if keyword in text:
                return keyword
        return "object"

    @staticmethod
    def _extract_location(text: str) -> str:
        """Simple heuristic to extract location name."""
        keywords = ["table", "shelf", "counter", "bin", "ground"]
        for keyword in keywords:
            if keyword in text:
                return keyword
        return "target_location"

