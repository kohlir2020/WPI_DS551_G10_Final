"""
Skill Combination Framework
Executes navigation and arm reaching skills on shared global environment
"""

import numpy as np
import os
import sys
from stable_baselines3 import PPO

# Add navigation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'navigation'))


class HRLNavigationSkill:
    """HRL navigation skill using high-level + low-level models"""
    
    def __init__(self, high_model_path, low_model_path, sim):
        """
        Args:
            high_model_path: Path to high-level PPO model
            low_model_path: Path to low-level PPO model
            sim: Shared Habitat simulator instance
        """
        self.name = "hrl_navigation"
        self.high_model = PPO.load(high_model_path)
        self.low_model = PPO.load(low_model_path)
        self.sim = sim
        self.agent = sim.get_agent(0)
        self.pathfinder = sim.pathfinder
        self.success_distance = 0.6  # meters
        
        # Setup HRL environment wrapper (using existing implementation)
        from navigation.hrl_highlevel_env import HRLHighLevelEnvImproved
        # from navigation.hrl_hac_env import HRLHACEnv  # DQN-based (not trained)
        self.hrl_env = HRLHighLevelEnvImproved(
            low_level_model_path=low_model_path,
            subgoal_distance=5.0,
            option_horizon=50,
            debug=False
        )
        # Increase max high-level steps for longer navigation
        self.hrl_env.max_highlevel_steps = 100  # Allow up to 5000 low-level steps
        
        # Replace HRL env's sim with shared sim
        self.hrl_env.sim = sim
        self.hrl_env.ll_env.sim = sim
        self.hrl_env.ll_env.agent = self.agent
        self.hrl_env.pathfinder = self.pathfinder
    
    def reset_with_goal(self, goal_position):
        """Reset skill with specific goal"""
        # Set goal in low-level env
        self.hrl_env.ll_env.goal_position = np.array(goal_position, dtype=np.float32)
        self.hrl_env.main_goal = np.array(goal_position, dtype=np.float32)
        
        # Reset the HRL environment to initialize state properly
        obs, _ = self.hrl_env.reset()
        return obs
    
    def step(self, obs):
        """Execute one high-level step (which runs up to 50 low-level actions)"""
        action, _ = self.high_model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, info = self.hrl_env.step(action)
        return next_obs, reward, done or truncated, info
    
    def is_complete(self, obs):
        """Check if navigation goal reached"""
        distance = obs[0]
        return distance < self.success_distance


class ArmReachingSkill:
    """Arm reaching skill - move arm to target"""
    
    def __init__(self, model_path, sim):
        """
        Args:
            model_path: Path to trained arm reaching PPO model
            sim: Shared Habitat simulator instance
        """
        self.name = "arm_reaching"
        self.model = PPO.load(model_path)
        self.sim = sim
        self.success_distance = 0.15  # meters (gripper trigger distance)
        
    def reset_with_goal(self, goal_params):
        """Reset skill with target position"""
        # TODO: Setup arm environment with goal
        pass
    
    def step(self, obs):
        """Execute one arm control step"""
        action, _ = self.model.predict(obs, deterministic=True)
        # TODO: Step arm environment
        return obs, 0.0, False, {}
    
    def is_complete(self, obs):
        """Check if arm goal reached"""
        distance = obs[0]
        return distance < self.success_distance


def execute_skill(skill, goal_params, max_steps=500, visualizer=None):
    """
    Execute a skill until complete or timeout
    
    Args:
        skill: Skill instance (HRLNavigationSkill or ArmReachingSkill)
        goal_params: Goal parameters (e.g., {"target": [x, y, z]})
        max_steps: Maximum steps before timeout
        visualizer: Optional visualizer for RGB rendering
        
    Returns:
        (success, info): Whether skill succeeded and final info dict
    """
    print(f"\n▶ Executing skill: {skill.name}")
    print(f"  Goal: {goal_params}")
    
    # Reset skill with goal
    if skill.name == "hrl_navigation":
        obs = skill.reset_with_goal(goal_params["target"])
        # Print start position
        agent_pos = skill.agent.get_state().position
        print(f"  Start: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}]")
        print(f"  Initial distance: {obs[0]:.2f}m")
    else:
        obs = skill.reset_with_goal(goal_params)
    
    step = 0
    total_reward = 0.0
    done = False
    
    while not done and step < max_steps:
        # Execute step
        obs, reward, done, info = skill.step(obs)
        total_reward += reward
        step += 1
        
        # Check completion
        if skill.is_complete(obs):
            done = True
            success = True
        
        # Visualization
        if visualizer and step % 1 == 0:  # Every 5 steps
            visualizer.render_frame(skill.name, step, obs)
        
        # Print progress with position
        if skill.name == "hrl_navigation" and step % 10 == 0:
            agent_pos = skill.agent.get_state().position
            print(f"  Step {step}/{max_steps}: pos=[{agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f}], "
                  f"dist={obs[0]:.2f}m, reward={total_reward:.2f}")
        elif step % 10 == 0:
            print(f"  Step {step}/{max_steps}: dist={obs[0]:.2f}m, reward={total_reward:.2f}")
    
    success = skill.is_complete(obs)
    final_info = {
        "success": success,
        "steps": step,
        "final_distance": obs[0],
        "total_reward": total_reward
    }
    
    print(f"  {'✓ SUCCESS' if success else '✗ FAILED'} after {step} steps")
    print(f"  Final distance: {obs[0]:.2f}m")
    
    return success, final_info


# OLD CODE (separate environment instances):
# class SkillBase:
#     """Base class for skills"""
#     
#     def __init__(self, name, model_path):
#         self.name = name
#         self.model = PPO.load(model_path)
#         self.current_obs = None
#         
#     def reset(self, obs):
#         """Reset skill with initial observation"""
#         self.current_obs = obs
#         
#     def step(self, action_spec=None):
#         """Execute one step of the skill"""
#         action, _ = self.model.predict(self.current_obs, deterministic=True)
#         return action
#     
#     def is_complete(self, obs, info):
#         """Check if skill has completed its goal"""
#         raise NotImplementedError
# 
# 
# class NavigationSkill(SkillBase):
#     """Navigation skill - move agent to goal"""
#     
#     def __init__(self, model_path="models/lowlevel_curriculum_250k"):
#         super().__init__("navigation", model_path)
#         self.success_distance = 0.5  # meters
#         
#     def is_complete(self, obs, info):
#         """Success when distance < threshold"""
#         distance = obs[0]
#         return distance < self.success_distance
# 
# 
# class FetchSkill(SkillBase):
#     """Fetch navigation skill - move fetch base to goal"""
#     
#     def __init__(self, model_path="models/fetch/fetch_nav_250k_final"):
#         super().__init__("fetch_navigation", model_path)
#         self.success_distance = 0.5  # meters
#         
#     def is_complete(self, obs, info):
#         """Success when distance < threshold"""
#         distance = obs[0]
#         return distance < self.success_distance
# 
# 
# class SkillExecutor:
#     """
#     Executes a sequence of skills
#     
#     Example:
#         executor = SkillExecutor()
#         executor.add_skill(NavigationSkill())
#         executor.add_skill(FetchSkill())
#         
#         # In main loop:
#         obs, info = env.reset()
#         for skill in executor.skills:
#             obs, success = executor.execute_skill(skill, obs, env, max_steps=500)
#             print(f"{skill.name}: {'SUCCESS' if success else 'FAILED'}")
#     """
#     
#     def __init__(self):
#         self.skills = []
#         self.history = []
#         
#     def add_skill(self, skill):
#         """Add a skill to executor"""
#         self.skills.append(skill)
#         print(f"✓ Added skill: {skill.name}")
#         
#     def execute_skill(self, skill, initial_obs, env, max_steps=500):
#         """
#         Execute a single skill
#         
#         Args:
#             skill: Skill instance
#             initial_obs: Initial observation
#             env: Environment (SimpleNavigationEnv or FetchNavigationEnv)
#             max_steps: Maximum steps before timeout
#             
#         Returns:
#             final_obs: Observation after skill execution
#             success: Whether skill completed successfully
#         """
#         skill.reset(initial_obs)
#         obs = initial_obs
#         
#         print(f"\n>>> Executing {skill.name}...")
#         
#         for step in range(max_steps):
#             # Get action from skill's policy
#             action = skill.step()
#             
#             # Execute in environment
#             obs, reward, done, truncated, info = env.step(action)
#             
#             # Check skill completion
#             if skill.is_complete(obs, info):
#                 print(f"✓ {skill.name} completed in {step+1} steps")
#                 self.history.append({
#                     'skill': skill.name,
#                     'steps': step + 1,
#                     'success': True,
#                     'distance': obs[0]
#                 })
#                 return obs, True
#             
#             if truncated:
#                 print(f"✗ {skill.name} timeout after {max_steps} steps (distance: {obs[0]:.2f}m)")
#                 self.history.append({
#                     'skill': skill.name,
#                     'steps': step + 1,
#                     'success': False,
#                     'distance': obs[0]
#                 })
#                 return obs, False
#         
#         print(f"✗ {skill.name} failed")
#         self.history.append({
#             'skill': skill.name,
#             'steps': max_steps,
#             'success': False,
#             'distance': obs[0]
#         })
#         return obs, False
#     
#     def execute_sequence(self, env, skill_sequence=None, max_steps_per_skill=500):
#         """
#         Execute multiple skills in sequence
#         
#         Args:
#             env: Environment
#             skill_sequence: List of skill indices or names. If None, uses self.skills
#             max_steps_per_skill: Max steps per skill
#             
#         Returns:
#             success: Whether entire sequence completed
#             history: List of skill execution results
#         """
#         if skill_sequence is None:
#             skill_sequence = list(range(len(self.skills)))
#         
#         obs, _ = env.reset()
#         sequence_success = True
#         
#         print(f"\n{'='*70}")
#         print(f"EXECUTING SKILL SEQUENCE: {[self.skills[i].name for i in skill_sequence]}")
#         print(f"{'='*70}")
#         
#         for skill_idx in skill_sequence:
#             skill = self.skills[skill_idx]
#             obs, success = self.execute_skill(skill, obs, env, max_steps_per_skill)
#             
#             if not success:
#                 sequence_success = False
#                 print(f"\n⚠ Sequence failed at {skill.name}")
#                 break
#         
#         return sequence_success, self.history
#     
#     def print_summary(self):
#         """Print execution summary"""
#         if not self.history:
#             print("No executions yet")
#             return
#             
#         print(f"\n{'='*70}")
#         print(f"EXECUTION SUMMARY")
#         print(f"{'='*70}\n")
#         
#         total_steps = 0
#         successful = 0
#         
#         for record in self.history:
#             status = "✓ SUCCESS" if record['success'] else "✗ FAILED"
#             print(f"{record['skill']:20} | {status:10} | "
#                   f"Steps: {record['steps']:4} | Distance: {record['distance']:6.2f}m")
#             total_steps += record['steps']
#             if record['success']:
#                 successful += 1
#         
#         success_rate = (successful / len(self.history)) * 100 if self.history else 0
#         print(f"\n{'='*70}")
#         print(f"Total Skills: {len(self.history)}")
#         print(f"Successful: {successful} ({success_rate:.1f}%)")
#         print(f"Total Steps: {total_steps}")
#         print(f"{'='*70}\n")


def demo_skill_executor():
    """Demo: Execute navigation then fetch in sequence"""

    from navigation.simple_navigation_env import SimpleNavigationEnv
    from navigation.hrl_highlevel_env import HRLHighLevelEnvImproved
    from arm.fetch_navigation_env import FetchNavigationEnv
    
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
