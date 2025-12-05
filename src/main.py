"""
Main execution script for LLM-planned multi-task robot control
Loads scene, gets plan, executes navigation (+ arm) tasks with visualization
"""
import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from shared.scene_manager import create_fetch_scene
from planner.llm_planner import get_hardcoded_plan  # , TaskPlanner for LLM
from skill_executor import HRLNavigationSkill, execute_skill


# Test goals (simple coordinates)
GOALS = {
    "drawer": [10.0, 0.0, 5.0],
    "table": [15.0, 0.0, 8.0],
    "shelf": [8.0, 0.0, 12.0]
}


def get_hardcoded_plan():
    """Hard-coded plan for testing (no LLM dependency)"""
    return [
        {
            "skill": "navigate",
            "params": {"target": [10.0, 0.0, 5.0]}  # drawer
        },
        {
            "skill": "navigate", 
            "params": {"target": [15.0, 0.0, 8.0]}  # table
        }
    ]


class TaskVisualizer:
    """Handles RGB visualization during task execution"""
    
    def __init__(self, sim, enable_video=False):
        self.sim = sim
        self.enable_video = enable_video
        self.video_frames = []
    
    def render_frame(self, skill_name, step, obs):
        """Render and display current frame"""
        observations = self.sim.get_sensor_observations()
        rgb = observations.get("color_sensor")
        
        if rgb is not None:
            # Convert RGB to BGR for OpenCV
            rgb = rgb[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Add overlay
            frame = self._add_overlay(bgr, skill_name, step, obs[0])
            
            if self.enable_video:
                self.video_frames.append(frame)
            
            # Display
            cv2.imshow("Multi-Task Execution", frame)
            cv2.waitKey(1)
    
    def _add_overlay(self, frame, skill_name, step, distance):
        """Add info overlay to frame"""
        frame = frame.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Skill: {skill_name}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Step: {step}", (10, 50), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 75), font, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_video(self, filename):
        """Save recorded frames as video"""
        if not self.video_frames:
            print("No frames to save")
            return
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        height, width = self.video_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10, (width, height))
        
        for frame in self.video_frames:
            out.write(frame)
        
        out.release()
        print(f"✓ Video saved: {filename}")
    
    def close(self):
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Multi-task robot execution with LLM planning")
    parser.add_argument("--goal", type=str, default="drawer", choices=list(GOALS.keys()),
                       help="Goal location to navigate to")
    parser.add_argument("--save-video", action="store_true",
                       help="Save video of execution")
    parser.add_argument("--low-level-model", type=str, 
                       default="src/navigation/models/lowlevel_curriculum_250k.zip",
                       help="Path to low-level navigation model")
    parser.add_argument("--high-level-model", type=str,
                       default="src/navigation/models/hl_improved/highlevel_improved_final.zip",
                       help="Path to high-level navigation model")
    # parser.add_argument("--arm-model", type=str,
    #                    default="models/arm_reaching.zip",
    #                    help="Path to arm reaching model")
    # parser.add_argument("--use-llm", action="store_true",
    #                    help="Use LLM planner instead of hard-coded plan")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-TASK ROBOT EXECUTION")
    print("="*70)
    print(f"Goal: {args.goal} -> {GOALS[args.goal]}")
    print()
    
    # Create scene with physics + RGB sensor
    print("Setting up scene...")
    sim, agent, pathfinder = create_fetch_scene(
        enable_physics=True,
        add_rgb_sensor=True
    )
    print("✓ Scene ready (Skokloster Castle + Fetch robot + physics)")
    
    # Set random start position
    start_pos = pathfinder.get_random_navigable_point()
    agent_state = agent.get_state()
    agent_state.position = start_pos
    agent.set_state(agent_state)
    print(f"✓ Start position: {start_pos}")
    
    # Load trained models
    print("\nLoading models...")
    try:
        nav_skill = HRLNavigationSkill(
            high_model_path=args.high_level_model,
            low_model_path=args.low_level_model,
            sim=sim
        )
        print("✓ HRL navigation skill loaded")
    except FileNotFoundError as e:
        print(f"✗ Failed to load models: {e}")
        print("Train navigation models first:")
        print("  python src/navigation/train_low_level_nav.py")
        print("  python src/navigation/train_high_level_improved.py")
        sim.close()
        return
    
    # Get task plan
    print("\nGenerating plan...")
    goal_pos = GOALS[args.goal]
    
    # USE HARD-CODED PLAN FOR NOW:
    plan = get_hardcoded_plan()
    
    # TO USE LLM (requires OPENAI_API_KEY):
    # if args.use_llm:
    #     planner = TaskPlanner()
    #     plan = planner.plan(start_pos, goal_pos)
    # else:
    #     plan = get_hardcoded_plan()
    
    print(f"✓ Plan: {len(plan)} tasks")
    for i, task in enumerate(plan):
        print(f"  {i+1}. {task['skill']}: {task['params']}")
    
    # Execute tasks
    print("\n" + "="*70)
    print("EXECUTING TASKS")
    print("="*70)
    
    visualizer = TaskVisualizer(sim, enable_video=args.save_video)
    results = []
    
    for i, task in enumerate(plan):
        print(f"\n--- Task {i+1}/{len(plan)}: {task['skill']} ---")
        
        if task["skill"] == "navigate":
            success, info = execute_skill(
                nav_skill,
                task["params"],
                max_steps=500,
                visualizer=visualizer
            )
            results.append({"task": task["skill"], "success": success, "info": info})
            
            if not success:
                print(f"✗ Navigation failed, aborting remaining tasks")
                break
        
        elif task["skill"] == "reach_arm":
            print("⚠ Arm reaching not implemented yet (skipping)")
            # arm_success, arm_info = execute_skill(arm_skill, task["params"], max_steps=200)
            # results.append({"task": task["skill"], "success": arm_success, "info": arm_info})
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    successful = sum(1 for r in results if r["success"])
    print(f"Tasks completed: {successful}/{len(results)}")
    for i, r in enumerate(results):
        status = "✓" if r["success"] else "✗"
        print(f"  {status} Task {i+1} ({r['task']}): "
              f"{r['info']['steps']} steps, "
              f"final dist={r['info']['final_distance']:.2f}m")
    print("="*70)
    
    # Save video if requested
    if args.save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"videos/main_execution_{args.goal}_{timestamp}.mp4"
        visualizer.save_video(video_name)
    
    # Cleanup
    visualizer.close()
    sim.close()
    print("\n✓ Done")


if __name__ == "__main__":
    main()

# OLD SKELETON CODE (kept for reference):
# ============================================================================
# from skillTask import SkillTask
# 
# def make_global_env():
#     """
#     This function will initialize the environment/simulator from Habitat
#     It sets up the observation space, the total action space (not all tasks will take all actions, 
#     but all tasks will be limited to actions from this set), and a definition for done.
#     """
#     pass
# 
# def load_all_skills() -> dict[tuple, SkillTask]:
#     """
#     This function will initialize all the trained skills (with pre-trained models)
#     Each skill has a description of how its executed for the planning agent to use as context.
#     """
#     pass
# 
# def extract_state():
#     pass
# 
# def get_planner(planner_type):
#     """
#     Sets up the planner agent (VLM)
#     """
#     return None
# 
# def check_global_goal(Env, goal_spec, obs, info) -> bool:
#     """
#     Checks if the goal is completed
#     """
#     pass
# 
# def main(goal_spec, planner_type):
#     env = make_global_env()         # Habitat kitchen with Fetch+fridge
#     obs, info = env.reset()
# 
#     # Load all skill policies (maybe multiple algos per skill)
#     skills = load_all_skills()
# 
#     planner = get_planner(planner_type)
# 
#     done = False
#     while not done:
#         state_repr = extract_state(env, obs, info)
#         plan_or_call = planner(state_repr, goal_spec, skills)
# 
#         # easiest: planner returns *one* next skill call:
#         skill_name, algo_name, args = plan_or_call
#         skill = skills[(skill_name, algo_name)]
# 
#         skill.reset_runtime(env, goal=args)
#         skill_done = False
#         while not skill_done:
#             skill_done, (obs, info) = skill.step_runtime(env)
# 
#         done = check_global_goal(env, goal_spec, obs, info)

