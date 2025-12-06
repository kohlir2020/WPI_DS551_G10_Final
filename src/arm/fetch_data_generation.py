#! /usr/bin/python3
import numpy as np
from rrt_algorithms.rrt import rrt_connect
from rrt_algorithms.search_space.search_space import SearchSpace
from fetch_arm_reaching_env import FetchArmReachingEnv

default_pose = np.array([
            0.0,      # shoulder_pan_joint
            1.3,      # shoulder_lift_joint  
            0.0,      # upperarm_roll_joint
            -2.2,     # elbow_flex_joint
            0.0,      # forearm_roll_joint
            2.0,      # wrist_flex_joint
            0.0       # wrist_roll_joint
        ], dtype=np.float32)

def gen_collision_fn(env):
    def collision_fn(x_a, x_b, r):
        # take substeps between x_a and x_b with step size r
        check_pts = np.arange(x_a, x_b, r)
        for pt in check_pts:
            if env.check_collision(pt) == True:
                return True
        return False
    return collision_fn

def generate_fetch_data(env, num_samples):
    collision_fn = gen_collision_fn(env)
    data = []

    x_space = [(-np.pi, np.pi) for _ in range(7)]
    x_space = SearchSpace(x_space)
    q = [0.1 for _ in range(7)]
    max_samples = 100000
    res = 0.05
    prc = 0.1

    for _ in range(num_samples):
        print(f"Generating sample {_+1}/{num_samples}")
        while True:
            start_pose = np.random.uniform(
                low=[-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
                high=[ np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi]
            )
            if not env.check_collision(start_pose):
                break
        while True:
            goal_pose = start_pose = np.random.uniform(
                low=[-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
                high=[ np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi]
            )
            if not env.check_collision(goal_pose):
                break
        
        print(f"  Start pose: {start_pose}")
        print(f"  Goal pose:  {goal_pose}")
        
        rrt_planner = rrt_connect.RRTConnect(
            X=x_space,
            q=q,
            x_init=tuple(start_pose),
            x_goal=tuple(goal_pose),
            max_samples=max_samples,
            r=res,
            prc=prc,
            collision_fn=collision_fn
        )
        
        path = rrt_planner.rrt_connect()
        
        if path is not None:
            data.append({
                'start': start_pose,
                'goal': goal_pose,
                'path': path
            })
    
    return data

def main():
    env = FetchArmReachingEnv()
    num_samples = 100
    data = generate_fetch_data(env, num_samples)
    
    # Save data to file
    import pickle
    with open('fetch_arm_reaching_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Generated {len(data)} samples of fetch arm reaching data.")

if __name__ == "__main__":
    main()



