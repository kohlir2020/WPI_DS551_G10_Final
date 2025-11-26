# this is our main entry point into the code
from skillTask import SkillTask

def make_global_env():
    """
    This function will initialize the environment/simulator from Habitat
    It sets up the observation space, the total action space (not all tasks will take all actions, 
    but all tasks will be limited to actions from this set), and a definition for done.
    """
    pass

def load_all_skills() -> dict[tuple, SkillTask]:
    """
    This function will initialize all the trained skills (with pre-trained models)
    Each skill has a description of how its executed for the planning agent to use as context.
    """
    pass

def extract_state():
    pass

def get_planner(planner_type):
    """
    Sets up the planner agent (VLM)
    """
    return None

def check_global_goal(Env, goal_spec, obs, info) -> bool:
    """
    Checks if the goal is completed
    """
    pass

def main(goal_spec, planner_type):
    env = make_global_env()         # Habitat kitchen with Fetch+fridge
    obs, info = env.reset()

    # Load all skill policies (maybe multiple algos per skill)
    skills = load_all_skills()

    planner = get_planner(planner_type)

    done = False
    while not done:
        state_repr = extract_state(env, obs, info)
        plan_or_call = planner(state_repr, goal_spec, skills)

        # easiest: planner returns *one* next skill call:
        skill_name, algo_name, args = plan_or_call
        skill = skills[(skill_name, algo_name)]

        skill.reset_runtime(env, goal=args)
        skill_done = False
        while not skill_done:
            skill_done, (obs, info) = skill.step_runtime(env)

        done = check_global_goal(env, goal_spec, obs, info)
