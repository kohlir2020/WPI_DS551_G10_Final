import torch
from agent import DQNAgent
from environment import HabitatEnv

def train(num_episodes=1000):
    # Initialize environment and agent
    env = HabitatEnv("configs/habitat_config.yaml")
    agent = DQNAgent(env.observation_size, env.action_size)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state)
            agent.train()
            
            state = next_state
            total_reward += reward
            
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
            
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    train()