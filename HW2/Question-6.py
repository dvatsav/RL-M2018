
# coding: utf-8

# In[1]:


from copy import deepcopy
import numpy as np


# In[2]:


discount_factor = 1
n = 4
action_names = {'-1' : 'g', '0' : 'l', '1' : 'u', '2' : 'r', '3' : 'd'}
actions = [(0, -1), (-1, 0), (0, 1), (1,0)]
values = np.zeros((n, n))
policy = np.zeros((n, n), dtype=int)
policy[0][0] = -1
policy[n-1][n-1] = -1
policy_multiple = {(0,0):[], (n-1,n-1):[]} # Used to store the multiple actions if they return the same reward


# In[3]:


"""
* Policy Iteration Method
"""

def get_state(i, j, action):
    if i + action[0] < 0 or i + action[0] >= n or j + action[1] < 0 or j + action[1] >= n: # Check bounds after checking for A and B
        x,y = i,j
    else:
        x,y = i + action[0], j + action[1]
    return (x,y)

def evaluate_policy():
    
    theta = 2
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                old_value_s = values[i, j]
                if policy[i, j] != -1:
                    action = actions[policy[i,j]]
                    reward = -1.0
                    x, y = get_state(i, j, action)
                    values[i, j] = (reward + discount_factor * (values[x, y]))
                
                delta = max(delta, np.abs(old_value_s - values[i, j]))
        if delta < theta:
            break
    return values
        
def improve_policy():
    stable = True
    for i in range(n):
        for j in range(n):
            returns = np.zeros(len(actions))
            if policy[i, j] != -1:
                old_policy = policy[i, j]
                max_return = -np.inf
                best_action = actions[old_policy]
                for ind, action in enumerate(actions):
                    reward = -1.0
                    x, y = get_state(i, j, action)
                    expected_return = (reward + discount_factor * (values[x, y]))
                    returns[ind] = expected_return
                
                # We only consider the first max as the policy for the next iteration. Since we consider actions in a fixed order
                # this will always have the same value. This prevents infinite loops during iteration. However, we also store
                # all possible max actions in policy multiple
                policy[i, j] = np.argmax(returns)
                sorted_returns = np.argsort(returns)
                best_return = np.max(returns)
                best_actions = sorted_returns[len(actions) - (list(returns)).count(best_return):]
                best_actions = list(map(lambda x: action_names[str(x)], best_actions))
                policy_multiple[(i, j)] = best_actions

                if policy[i, j] != old_policy:
                    stable = False
    return stable


# In[4]:


it = 0
while True:
    it += 1
    values = evaluate_policy()
    stable = improve_policy()
    print ("Iteration", it)
    print ("-----------------------------")
    print ("Values")
    print (values)
    print ("-----------------------------")
    print ("Policy")
    for i in range(n):
        for j in range(n):
            print (policy_multiple[(i, j)], end = " ")
        print ("")
    print ("-----------------------------")
    if stable:
       break 


# In[5]:


"""
* Print Optimal Values
"""
values


# In[6]:


"""
* Print Optimal Policy
"""
for i in range(n):
    for j in range(n):
        print (policy_multiple[(i, j)], end = " ")
    print ("")


# In[7]:


"""
* Value Iteration
"""

values = np.zeros((n, n))
policy = np.zeros((n, n), dtype=int)
policy[0][0] = -1
policy[n-1][n-1] = -1
policy_multiple = {(0,0):[], (n-1,n-1):[]}


# In[8]:


theta = 1e-4
it = 0
while True:
    delta = 0
    it += 1
    for i in range(n):
        for j in range(n):
            returns = np.zeros(len(actions))
            if policy[i, j] != -1:
                old_value = values[i, j]
                max_return = -np.inf
                best_action = actions[policy[i, j]]
                
                for ind, action in enumerate(actions):
                    reward = -1.0
                    if i + action[0] < 0 or i + action[0] >= n or j + action[1] < 0 or j + action[1] >= n: # Check bounds after checking for A and B
                        x,y = i,j
                    else:
                        x,y = i + action[0], j + action[1]
                    expected_return = (reward + discount_factor * (values[x, y]))
                    returns[ind] = expected_return
                        
                values[i, j] = np.max(returns)
                policy[i, j] = np.argmax(returns)
                sorted_returns = np.argsort(returns)
                best_return = np.max(returns)
                best_actions = sorted_returns[len(actions) - (list(returns)).count(best_return):]
                best_actions = list(map(lambda x: action_names[str(x)], best_actions))
                policy_multiple[(i, j)] = best_actions
                
                delta = max(delta, np.abs(old_value - values[i, j]))
    print ("Iteration", it)
    print ("---------------------------")
    print ("Delta -", delta)
    print ("Values")
    print (values)
    print ("---------------------------")
    print ("Policy")
    for i in range(n):
        for j in range(n):
            print (policy_multiple[(i, j)], end = " ")
        print ("")
    print ("---------------------------")
    if delta < theta:
        break


# In[9]:


"""
* Optimal Values
"""
values


# In[10]:


"""
* Optimal Policy
"""
for i in range(n):
    for j in range(n):
        print (policy_multiple[(i, j)], end = " ")
    print ("")

