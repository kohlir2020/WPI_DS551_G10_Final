"""
Skill Combination Framework
Combines navigation and fetch as individual skills to create multi-task behavior
This is a bridge between individual low-level skills and the high-level executor
"""

import numpy as np
from stable_baselines3 import PPO
from src.simple_navigation_env import SimpleNavigationEnv
from src.arm.fetch_navigation_env import FetchNavigationEnv
import os


class SkillBase:
    """Base class for skills"""
    
    def __init__(self, name, model_path):
        self.name = name
        self.model = PPO.load(model_path)
        self.current_obs = None
        
    def reset(self, obs):
        """Reset skill with initial observation"""
        self.current_obs = obs
        
    def step(self, action_spec=None):
        """Execute one step of the skill"""
        action, _ = self.model.predict(self.current_obs, deterministic=True)
        return action
    
    def is_complete(self, obs, info):
        """Check if skill has completed its goal"""
        raise NotImplementedError


class NavigationSkill(SkillBase):
    """Navigation skill - move agent to goal"""
    
    def __init__(self, model_path="models/lowlevel_curriculum_250k"):
        super().__init__("navigation", model_path)
        self.success_distance = 0.5  # meters
        
    def is_complete(self, obs, info):
        """Success when distance < threshold"""
        distance = obs[0]
        return distance < self.success_distance


class FetchSkill(SkillBase):
    """Fetch navigation skill - move fetch base to goal"""
    
    def __init__(self, model_path="models/fetch/fetch_nav_250k_final"):
        super().__init__("fetch_navigation", model_path)
        self.success_distance = 0.5  # meters
        
    def is_complete(self, obs, info):
        """Success when distance < threshold"""
        distance = obs[0]
        return distance < self.success_distance


class SkillExecutor:
    """
    Executes a sequence of skills
    
    Example:
        executor = SkillExecutor()
        executor.add_skill(NavigationSkill())
        executor.add_skill(FetchSkill())
        
        # In main loop:
        obs, info = env.reset()
        for skill in executor.skills:
            obs, success = executor.execute_skill(skill, obs, env, max_steps=500)
            print(f"{skill.name}: {'SUCCESS' if success else 'FAILED'}")
    """
    
    def __init__(self):
        self.skills = []
        self.history = []
        
    def add_skill(self, skill):
        """Add a skill to executor"""
        self.skills.append(skill)
        print(f"✓ Added skill: {skill.name}")
        
    def execute_skill(self, skill, initial_obs, env, max_steps=500):
        """
        Execute a single skill
        
        Args:
            skill: Skill instance
            initial_obs: Initial observation
            env: Environment (SimpleNavigationEnv or FetchNavigationEnv)
            max_steps: Maximum steps before timeout
            
        Returns:
            final_obs: Observation after skill execution
            success: Whether skill completed successfully
        """
        skill.reset(initial_obs)
        obs = initial_obs
        
        print(f"\n>>> Executing {skill.name}...")
        
        for step in range(max_steps):
            # Get action from skill's policy
            action = skill.step()
            
            # Execute in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Check skill completion
            if skill.is_complete(obs, info):
                print(f"✓ {skill.name} completed in {step+1} steps")
                self.history.append({
                    'skill': skill.name,
                    'steps': step + 1,
                    'success': True,
                    'distance': obs[0]
                })
                return obs, True
            
            if truncated:
                print(f"✗ {skill.name} timeout after {max_steps} steps (distance: {obs[0]:.2f}m)")
                self.history.append({
                    'skill': skill.name,
                    'steps': step + 1,
                    'success': False,
                    'distance': obs[0]
                })
                return obs, False
        
        print(f"✗ {skill.name} failed")
        self.history.append({
            'skill': skill.name,
            'steps': max_steps,
            'success': False,
            'distance': obs[0]
        })
        return obs, False
    
    def execute_sequence(self, env, skill_sequence=None, max_steps_per_skill=500):
        """
        Execute multiple skills in sequence
        
        Args:
            env: Environment
            skill_sequence: List of skill indices or names. If None, uses self.skills
            max_steps_per_skill: Max steps per skill
            
        Returns:
            success: Whether entire sequence completed
            history: List of skill execution results
        """
        if skill_sequence is None:
            skill_sequence = list(range(len(self.skills)))
        
        obs, _ = env.reset()
        sequence_success = True
        
        print(f"\n{'='*70}")
        print(f"EXECUTING SKILL SEQUENCE: {[self.skills[i].name for i in skill_sequence]}")
        print(f"{'='*70}")
        
        for skill_idx in skill_sequence:
            skill = self.skills[skill_idx]
            obs, success = self.execute_skill(skill, obs, env, max_steps_per_skill)
            
            if not success:
                sequence_success = False
                print(f"\n⚠ Sequence failed at {skill.name}")
                break
        
        return sequence_success, self.history
    
    def print_summary(self):
        """Print execution summary"""
        if not self.history:
            print("No executions yet")
            return
            
        print(f"\n{'='*70}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*70}\n")
        
        total_steps = 0
        successful = 0
        
        for record in self.history:
            status = "✓ SUCCESS" if record['success'] else "✗ FAILED"
            print(f"{record['skill']:20} | {status:10} | "
                  f"Steps: {record['steps']:4} | Distance: {record['distance']:6.2f}m")
            total_steps += record['steps']
            if record['success']:
                successful += 1
        
        success_rate = (successful / len(self.history)) * 100 if self.history else 0
        print(f"\n{'='*70}")
        print(f"Total Skills: {len(self.history)}")
        print(f"Successful: {successful} ({success_rate:.1f}%)")
        print(f"Total Steps: {total_steps}")
        print(f"{'='*70}\n")


def demo_skill_executor():
    """Demo: Execute navigation then fetch in sequence"""
    
    print("\n" + "="*70)
    print("SKILL COMBINATION DEMO")
    print("="*70)
    
    try:
        # Initialize environments (they use the same scene)
        print("\nInitializing environments...")
        nav_env = SimpleNavigationEnv()
        print("✓ Navigation environment loaded")
        
        fetch_env = FetchNavigationEnv()
        print("✓ Fetch environment loaded")
        
        # Create executor
        executor = SkillExecutor()
        
        # Add skills
        nav_skill = NavigationSkill(model_path="models/lowlevel_curriculum_250k")
        executor.add_skill(nav_skill)
        
        fetch_skill = FetchSkill(model_path="models/fetch/fetch_nav_250k_final")
        executor.add_skill(fetch_skill)
        
        # Execute navigation first
        print("\n" + "="*70)
        print("PHASE 1: NAVIGATION")
        print("="*70)
        obs, success_nav = executor.execute_skill(nav_skill, None, nav_env, max_steps=500)
        
        # Execute fetch navigation next
        print("\n" + "="*70)
        print("PHASE 2: FETCH NAVIGATION")
        print("="*70)
        obs, success_fetch = executor.execute_skill(fetch_skill, None, fetch_env, max_steps=500)
        
        # Summary
        executor.print_summary()
        
        if success_nav and success_fetch:
            print("\n✓✓✓ MULTI-SKILL EXECUTION SUCCESS ✓✓✓")
        else:
            print("\n✗ Some skills failed")
        
        nav_env.close()
        fetch_env.close()
        
    except FileNotFoundError as e:
        print(f"\n✗ Model not found: {e}")
        print("Make sure trained models exist:")
        print("  - models/lowlevel_curriculum_250k.zip")
        print("  - models/fetch/fetch_nav_250k_final.zip")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_skill_executor()
