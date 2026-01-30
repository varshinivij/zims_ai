"""
Episode process:
1. Reset the environment (robot starts in a default state)
2. Observe the current state (IMU, servos, LiDAR, etc.)
3. Select an action using an epsilon-greedy policy
4. Execute the action in the environment
5. Receive the next state, reward, and done flag
6. Store the transition in replay memory
7. Train the Q-network from replay memory
8. End episode when the robot falls or max steps are reached
9. Decay epsilon to reduce exploration over time
"""
import time
from deep_q_learning.deep_q import DQNAgent
from deep_q import DQNAgent
from robot_env import RobotEnv
import numpy as np
import random
from tqdm import tqdm


#epsilon should start fairly low ~0.3 and decay at a faster rate, but must never reach 0 in order to keep exploring
EPISODES = 1000
MIN_EPSILON = 0.05                      
EPSILON_DECAY = 0.995
epsilon = 1.0

ep_rewards = []

def episode_loop():
    agent = DQNAgent()
    env = RobotEnv() 

    current_state = env.reset()
    
    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
        #reset values 
        episode_reward = 0              #total reward tracked, measures performance of agent
        step = 1                        #keeps track of how many times an agent takes an action
        current_state = env.reset()     
        done = False
        
        while not done:
            # Choose action using epsilon-greedy strategy
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            
            # Apply action and observe result
            new_state, reward, done = env.step(action)

            episode_reward += reward

            # Store experience for replay
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            
            # Train the agent using replay memory
            agent.train(done, step)

            current_state = new_state
            step += 1

            ep_rewards.append(episode_reward)

        # Reduce exploration over time, but never stop completely
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)