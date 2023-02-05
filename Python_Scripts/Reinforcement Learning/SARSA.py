# SARSA RL
import numpy as np
import gym

env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def update(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + alpha * (target - predict)

# Training the learing agent
reward = 0

for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)
    
    while t < max_steps:
        env.render()
        state2, reward, done, info = env.step(action1)
        action2 = choose_action(state2)
        # Learning Q-value
        update(state1, state2, reward, action1, action2)
        
        state1 = state2
        action1 = action2
        
        t += 1
        reward += 1
        
        if done:
            break
        
print("Performance : ", reward / total_episodes)
print(Q)
        
        
    