# LLM-Planned Multi-Task Robot Control

Implementation of hierarchical reinforcement learning with LLM-based task planning for robotic manipulation in Habitat simulator (Skokloster Castle environment with Fetch robot).

See `Final_DS551_Paper.pdf` for full methodology and results.

## Quick Start

```bash
# Run with hardcoded plan
python src/main.py

# Run with LLM planning (requires OPENAI_API_KEY)
export OPENAI_API_KEY='your-key'
python src/main.py --goal "navigate to kitchen" --use-llm

# Save execution video
python src/main.py --save-video
```

## Implementation Map

**Navigation (HRL):**
- Environment: `src/navigation/simple_navigation_env.py` (low-level)
- HRL Wrapper: `src/navigation/hrl_highlevel_env.py` (high-level manager)
- Models: `src/navigation/models/lowlevel_curriculum_250k.zip`, `highlevel_improved_final.zip`

**Arm Reaching:**
- Environment: `src/arm/habitat_arm_reaching_env.py`
- Models: `logs/simple_arm/cartesian_ppo_*/final_ppo.zip`

**Planning & Execution:**
- LLM Planner: `src/planner/llm_planner.py` (GPT-4o with structured output)
- Skill Executor: `src/skill_executor.py` (manages skill sequencing)
- Scene Manager: `src/shared/scene_manager.py` (Habitat simulator setup)
- Main Entry: `src/main.py`

## Requirements

- Habitat-Sim & Habitat-Lab (included in `habitat-sim/`, `habitat-lab/`)
- Python 3.9+, PyTorch, Stable-Baselines3, OpenAI API (optional)