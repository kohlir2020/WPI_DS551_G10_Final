## Summary

**Core Components:**
1. **`Environment`** - Wraps Habitat, handles initialization and stepping
2. **`TaskBase`** - Abstract class for RL-controlled tasks
3. **Concrete Tasks** - PickTask, PlaceTask, NavigateTask (each with their own RL model)
4. **`VLMPlanner`** - Uses OpenAI API to decompose natural language → task sequence
5. **`Executor`** - Runs task sequence, handles state handoff between tasks
6. **`main.py`** - Orchestrates everything

**Key Features:**
- Hand-coded subtasks (Pick, Place, Navigate)
- Task parameters passed from VLM planner  
- State handoff: previous task's final observation → next task's initial observation
- OpenAI API integration for VLM
- Heuristic fallback if VLM unavailable
- Visual rendering and metrics collection
- Modular and extensible for new tasks

```
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
```

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