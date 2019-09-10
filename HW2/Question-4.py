
# coding: utf-8

# In[1]:


from copy import deepcopy
import numpy as np
from scipy import optimize as opt


# In[2]:


gamma = 0.9
A = (0, 1)
B = (0, 3)
A1 = (4, 1)
B1 = (2, 3)
n = 5
actions = [(0, 1), (0, -1),(1, 0), (-1,0)]
action_names = {'0' : 'r', '1' : 'l', '2' : 'd', '3' : 'u'}


# In[3]:


"""
* System of Non - Linear Equations. We can modify this to a system of linear inequalities.
* We can model Bellman's Equation as a system of linear inequalities with 4 equations for each state, one equation for each 
* action. We can model this as V*_pi >= R_pi + gamma* P_pi * V*_pi, where P_pi is the state transition matrix. 
* We can then solve for v_pi* as a linear programming problem
* First we will initialize values that make up the MDP
"""

R_pi = np.zeros(4*n*n)
G = np.zeros((4*n*n, n*n)) # Coefficient matrix of V*_pi

# Model as a cost function instead of a reward function. We do this since we are minimizing while solving the LPP problem
def step_and_reward(i, j, action):
    if (i,j) == A:
        return (A1[0]*n+A1[1], -10)
    elif (i,j) == B:
        return (B1[0]*n+B1[1], -5)
    elif 0 <= i + action[0] < n and 0 <= j+action[1] < n:
        return ((i + action[0])*n + j+action[1], 0)
    else:
        return (i*n+j, 1)

# Set up rewards and transition matrix
for i in range(n):
    for j in range(n):
        for k in range(len(actions)):
            next_pos, reward = step_and_reward(i, j, actions[k])
            R_pi[4*(i*n+j) + k] += reward
            G[4*(i*n+j) + k, i*n+j] -= 1 # mimic an identiy matrix
            G[4*(i*n+j) + k, next_pos] += gamma


# In[4]:


G


# In[5]:


x = opt.linprog(np.ones(n*n), G, R_pi)
V_pi = np.round(x.x, 1).reshape(5,5)


# In[6]:


V_pi


# In[7]:


"""
* Optimal  Policy
"""
print ("Optimal Policy")
print ("------------------------")
for i in range(n):
    for j in range(n):
        returns = []
        for k in range(len(actions)):
            next_pos, reward = step_and_reward(i, j, actions[k])
            returns.append(V_pi.reshape(n*n)[next_pos])
        sorted_returns = np.argsort(returns)
        best_return = np.max(returns)
        best_actions = sorted_returns[len(actions) - returns.count(best_return):]
        best_actions = list(map(lambda x: action_names[str(x)], best_actions))
        print (best_actions, end=" ")
    print ()


# In[8]:


"""
* Iterative Value evaluation. Non-Linear equations solved in later cells
"""

grid = np.zeros((n, n))

delta = 1e-4
diff = 1000
while diff > delta:
    updated_grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            maxvalue = 0
            for action in actions: # four actions possible
                if (i,j) == A: # Give priority to A and B since all moves are valid here regardless of bounds
                    reward = 10.0
                    x,y = A1
                elif (i,j) == B:
                    reward = 5.0
                    x,y = B1
                elif i + action[0] < 0 or i + action[0] >= n or j + action[1] < 0 or j + action[1] >= n: # Check bounds after checking for A and B
                    reward = -1.0
                    x,y = i,j
                else:
                    reward = 0
                    x,y = i + action[0], j + action[1]
                # Bellman's equation
                maxvalue = max(maxvalue, (reward + gamma * (grid[x, y])))
            updated_grid[i, j] += maxvalue
    diff = np.abs(np.sum(np.subtract(updated_grid, grid)))
    grid = updated_grid


# In[9]:


print (grid)

