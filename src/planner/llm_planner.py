"""
LLM-based task planner using OpenAI
Generates task sequences from natural language goals
"""
import os
import json
from typing import List, Dict
from openai import OpenAI


class TaskPlanner:
    """LLM task planner using OpenAI with structured outputs"""
    
    def __init__(self):
        """Initialize OpenAI client. Requires OPENAI_API_KEY in environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def plan_from_goal(self, goal_description: str, start_pos: List[float]) -> List[Dict]:
        """
        Generate task plan from natural language goal
        
        Args:
            goal_description: Natural language goal (e.g., "navigate to the kitchen and pick up the cup")
            start_pos: Current robot position [x, y, z]
            
        Returns:
            List of task dicts: [{"skill": "navigate", "params": {"target": [x,y,z]}}, ...]
        """
        system_prompt = """You are a robot task planner. Given a goal description, generate a sequence of skills to execute.

Available skills:
1. navigate: Move robot base to a target position. Params: {"target": [x, y, z]}
2. reach_arm: Extend arm to reach an object at a height. Params: {"target_height": height_in_meters}

Return a JSON array of tasks. Example:
[{"skill": "navigate", "params": {"target": [4.1, 0.2, 6.6]}}, {"skill": "reach_arm", "params": {"target_height": 0.5}}]

IMPORTANT - Use these ACTUAL navigable coordinates from Skokloster Castle scene:
Navigation uses Skokloster Castle (actual navigable bounds: X[-9.76 to 8.57], Y[0.0-0.4], Z[0.97-25.60]):
- Navigable area center: around [-4.0, 0.2, 13.5] or [4.1, 0.2, 6.6]
- North area: [4.5, 0.2, 10.4]
- South area: [-1.5, 0.1, 20.3]
- West area: [-4.3, 0.2, 12.0]
- East area: [6.2, 0.2, 5.3]
- Always use Y height between 0.0-0.4 (floor level)
- Tables/obstacles exist, so stay within these tested points

Arm reaching uses realistic simulation (not Skokloster scene):
- Drawer handles: 0.5-0.7m
- Table objects: 0.6-0.8m
- Counter objects: 0.9-1.1m"""
        
        user_prompt = f"""Current robot position: {start_pos}
Goal: {goal_description}

Generate the task sequence as a JSON array."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle both {"tasks": [...]} and direct array formats
            if "tasks" in result:
                return result["tasks"]
            elif "plan" in result:
                return result["plan"]
            elif isinstance(result, list):
                return result
            else:
                # Assume result is a dict with the array somewhere
                for value in result.values():
                    if isinstance(value, list):
                        return value
                return []
                
        except Exception as e:
            print(f"⚠️  LLM planning failed: {e}")
            print(f"   Falling back to hardcoded plan")
            return None


def get_hardcoded_plan(goal_type="navigate_only"):
    """
    Hard-coded plans for testing (use this instead of LLM during development)
    Uses ACTUAL navigable coordinates from Skokloster Castle scene
    
    Args:
        goal_type: Type of goal task
        
    Returns:
        List of task dicts
    """
    plans = {
        "navigate_only": [
            {
                "skill": "navigate",
                "params": {"target": [4.1, 0.2, 6.6]}  # Actual navigable point
            }
        ],
        "navigate_and_reach": [
            {
                "skill": "navigate", 
                "params": {"target": [-4.0, 0.2, 13.5]}  # Actual navigable point
            },
            {
                "skill": "reach_arm",
                "params": {"target_height": 0.5}
            }
        ],
        "drawer_task": [
            {
                "skill": "navigate",
                "params": {"target": [4.5, 0.2, 10.4]}  # Actual navigable point
            },
            {
                "skill": "reach_arm",
                "params": {"target_height": 0.6}
            }
        ]
    }
    
    return plans.get(goal_type, plans["navigate_only"])
