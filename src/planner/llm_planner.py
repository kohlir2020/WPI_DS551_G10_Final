"""
LLM-based task planner using DSPY with GPT-4.1
Generates task sequences from start/goal positions
"""
import os
import json
#import dspy
from typing import List, Dict
from openai import OpenAI


class TaskPlanner:
    """LLM task planner using DSPY chain-of-thought"""
    
    def __init__(self):
        """Initialize DSPY with GPT-4.1. Requires OPENAI_API_KEY in environment."""
        #api_key = os.environ["OPENAI_API_KEY"]  # Hard fail if missing
        #self.lm = dspy.OpenAI(model="gpt-4.1", api_key=api_key)
        #dspy.settings.configure(lm=self.lm)
        
        # Define chain-of-thought signature
        #class PlanTasks(dspy.Signature):
        #    """Generate task sequence to reach goal from start position"""
        #    start_position = dspy.InputField(desc="Current robot position [x, y, z]")
        #    goal_position = dspy.InputField(desc="Target goal position [x, y, z]")
        #    available_skills = dspy.InputField(desc="Available skills: navigate, reach_arm")
        #    plan = dspy.OutputField(desc="JSON list of tasks with skill and params")
        #
        #self.predictor = dspy.ChainOfThought(PlanTasks)

        self.llm_client = OpenAI()

    
    def plan(self, start_pos, goal_pos):
        """
        Generate task plan using LLM
        
        Args:
            start_pos: Current position [x, y, z]
            goal_pos: Goal position [x, y, z]
            
        Returns:
            List of task dicts: [{"skill": "navigate", "params": {"target": [x,y,z]}}, ...]
        """
        #result = self.predictor(
        #    start_position=str(start_pos),
        #    goal_position=str(goal_pos),
        #    available_skills="navigate (move robot base), reach_arm (extend arm to target)"
        #)
        #
        ## Parse JSON from LLM output
        #plan_json = json.loads(result.plan)

        response = self.llm_client.responses.create(
            model="gpt-4.1",
            input="give me the plan" #TODO: FIX THIS!
        )

        return response.output_text


def get_hardcoded_plan(goal_type="navigate_only"):
    """
    Hard-coded plans for testing (use this instead of LLM during development)
    
    Args:
        goal_type: Type of goal task
        
    Returns:
        List of task dicts
    """
    plans = {
        "navigate_only": [
            {
                "skill": "navigate",
                "params": {"target": [10.0, 0.0, 5.0]}
            }
        ],
        "navigate_and_reach": [
            {
                "skill": "navigate", 
                "params": {"target": [10.0, 0.0, 5.0]}
            },
            {
                "skill": "reach_arm",
                "params": {"target_height": 0.5}
            }
        ],
        "drawer_task": [
            {
                "skill": "navigate",
                "params": {"target": [12.0, 0.0, 8.0]}
            },
            {
                "skill": "reach_arm",
                "params": {"target_height": 0.6}
            }
        ]
    }
    
    return plans.get(goal_type, plans["navigate_only"])
