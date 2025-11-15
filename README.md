This is the complete, production-ready file structure for your multi-task RL robot controller! Here's a summary of what you have:

## Summary

**Core Components:**
1. **`Environment`** - Wraps Habitat, handles initialization and stepping
2. **`TaskBase`** - Abstract class for RL-controlled tasks
3. **Concrete Tasks** - PickTask, PlaceTask, NavigateTask (each with their own RL model)
4. **`VLMPlanner`** - Uses OpenAI API to decompose natural language → task sequence
5. **`Executor`** - Runs task sequence, handles state handoff between tasks
6. **`main.py`** - Orchestrates everything

**Key Features:**
✓ Hand-coded subtasks (Pick, Place, Navigate)
✓ Task parameters passed from VLM planner  
✓ State handoff: previous task's final observation → next task's initial observation
✓ OpenAI API integration for VLM
✓ Heuristic fallback if VLM unavailable
✓ Visual rendering and metrics collection
✓ Modular and extensible for new tasks


I'll now create the complete file structure and implementation following your selections. Let me build out the modular program step by step.

Read WPI_DS551_G10_Final

Now I'll create the complete file structure for your multi-task RL robot controller system:

Read interactive_play.py, lines 1 to 100

Now I'll create the complete modular structure. I'll build all the necessary files:

Let me create the complete program structure. Since file editing isn't available, I'll provide the comprehensive file structure with code:

Let me check the end of the file:

Read interactive_play.py, lines 750 to 804

Perfect! Now I'll provide you with the complete file structure and code implementation. Since I can't create files directly, I'll output complete code blocks for each file you'll need to create:

Complete File Structure
Now here's each file's content:

1. src/__init__.py
2. src/environment.py (Habitat wrapper)
3. src/task_base.py (Abstract Task class)
4. src/tasks/__init__.py
5. src/tasks/task_registry.py (Task factory)
6. src/tasks/pick_task.py (Pick task implementation)
7. src/tasks/place_task.py (Place task implementation)
8. src/tasks/navigate_task.py (Navigation task implementation)
9. src/vlm_planner.py (VLM-based task decomposer)
10. src/executor.py (Execution orchestrator)
11. src/models/model_loader.py (Model checkpoint loading utility)
12. src/config/task_configs.py (Task parameter definitions)
13. src/config/__init__.py
14. src/main.py (Main entry point - THE ORCHESTRATOR)
15. requirements.txt
16. models/README.md (Model directory instructions)
Obtaining Models
For now, the system falls back to heuristic control if models are unavailable.
To integrate trained models:

Train models using habitat-baselines PPO or your own training code
Save model checkpoints: torch.save(model, "pick_model.pt")
Place in this directory
Update paths in src/tasks/pick_task.py, etc.
User Input
↓
[VLM Planner] → Decompose into task sequence
↓
[Executor] → Run tasks sequentially
↓
[Task 1] (Pick) → RL Policy 1
↓
[Task 2] (Place) → RL Policy 2
↓
[Task 3] (Navigate) → RL Policy 3
↓
[Visualization + Metrics]

## Quick Start
Run with default command:
`python src/main.py`
Run with custom command:
`python src/main.py --command "pick up the cube and move to the table"`
Interactive mode:
`python src/main.py --interactive`
With video output:
`python src/main.py --command "pick up the sphere" --save-video --video-path ./demo.mp4`

## How It Works
1. Environment Setup (environment.py)
Wraps Habitat environment initialization
Handles config loading and agent setup
Provides unified interface for stepping/rendering
2. Task Framework (task_base.py)
Abstract TaskBase class for all tasks
Each task loads an RL model and executes via step()
Tracks rewards, steps, and completion
3. Concrete Tasks (tasks/)
PickTask: Arm control for grasping
PlaceTask: Arm control for releasing
NavigateTask: Base velocity control
4. VLM Planner (vlm_planner.py)
Sends user command to OpenAI GPT
Parses response into task sequence
Falls back to heuristic matching if VLM unavailable
5. Executor (executor.py)
Runs task sequence sequentially
Passes observations between tasks
Collects metrics and renders visualization
6. Main (main.py)
Orchestrates entire pipeline
Handles command-line arguments
Displays results and statistics