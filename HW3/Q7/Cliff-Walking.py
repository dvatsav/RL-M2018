
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from random import random as rand, choice, randint
from tqdm import tqdm

get_ipython().magic('matplotlib inline')


# In[8]:


rows = 4
cols = 12
terminal_state = [rows-1, cols-1]
start_state = [rows-1, 0]
epsilon = 0.1
gamma = 1
alpha = 0.5
num_actions = 4
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] # Left Right Up Down
rewards = np.zeros((rows, cols)) - 1
rewards[rows-1, 1:cols-1] = -100
cliff = [[rows-1, i] for i in range(1, cols-1)]
# print (rewards)
# print (cliff)


# In[9]:


def get_action(state_action_values, state):
    possible_actions = state_action_values[state[0], state[1], :]
    best_action = choice([action for action, value in enumerate(possible_actions) if value == np.max(possible_actions)])
    if rand() > epsilon:
        return actions[best_action]
    else:
        return actions[randint(0, num_actions-1)]
    
def perform_action(state, action):
    new_state = [state[0] + action[0], state[1] + action[1]]
    if new_state[0] >= rows or new_state[0] < 0 or new_state[1] >= cols or new_state[1] < 0:
        new_state = state
    reward = rewards[new_state[0], new_state[1]]
    if new_state in cliff:
        new_state = start_state
    return new_state, reward


# In[10]:


def sarsa(state_action_values):
    state = start_state
    action = get_action(state_action_values, state)
    reward_sum = 0
    while state != terminal_state:
        next_state, reward = perform_action(state, action)
        reward_sum += reward
        next_action = get_action(state_action_values, next_state)
        #print ("Next action:", next_action)
        state_action_values[state[0], state[1], action] = state_action_values[state[0], state[1], action]             + alpha * (reward + gamma * state_action_values[next_state[0], next_state[1], next_action] - state_action_values[state[0], state[1], action])
        state = next_state
        action = next_action
        # print (state_action_values)
    
    return reward_sum
    
def q_learning(state_action_values):
    state = start_state
    reward_sum = 0
    while state != terminal_state:
        action = get_action(state_action_values, state)
        next_state, reward = perform_action(state, action)
        reward_sum += reward
        state_action_values[state[0], state[1], action] = state_action_values[state[0], state[1], action]             + alpha * (reward + gamma * np.max(state_action_values[next_state[0], next_state[1], :]) - state_action_values[state[0], state[1], action])
        state = next_state
        
    return reward_sum


# In[11]:


def plot_1():
    num_episodes = 500
    sarsa_sum_rewards = np.zeros(num_episodes)
    q_learning_sum_rewards = np.zeros(num_episodes)
    
    sarsa_state_action_values = np.zeros((rows, cols, num_actions))
    q_learning_state_action_values = np.zeros((rows, cols, num_actions))
    
    for i in tqdm(range(num_episodes)):
        sarsa_sum_rewards[i] = sarsa(sarsa_state_action_values)
        # print (sarsa_state_action_values)
        q_learning_sum_rewards[i] = q_learning(q_learning_state_action_values)
        
    fig = plt.figure()
    plt.plot(sarsa_sum_rewards, color='blue', label='Sarsa')
    plt.plot(q_learning_sum_rewards, color='red', label='Q-learning')
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.title("Cliff Walking")
    plt.legend()
    plt.show()


# In[12]:


plot_1()

