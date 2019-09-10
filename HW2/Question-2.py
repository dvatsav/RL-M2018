
# coding: utf-8

# In[1]:


from copy import deepcopy
import numpy as np


# In[2]:


gamma = 0.9
A = (0, 1)
B = (0, 3)
A1 = (4, 1)
B1 = (2, 3)
n = 5
action_prob = 0.25
actions = [(0, -1), (-1, 0), (0, 1), (1,0)]


# In[3]:


"""
* System of Linear Equations
* We can model Bellman's Equation as a system of linear equations with an equation for each state. We can model this as
* V_pi = R_pi + gamma* P_pi * V_pi, where P_pi is the state transition matrix. We can find V_pi as (1-gamma*P_pi)^-1 * R_pi
* First we will initialize values that make up the MDP
"""

R_pi = np.zeros((n*n, 1))
P_pi = np.zeros((n*n, n*n))
identity = np.identity(n*n, dtype=float)

# Set up rewards
for i in range(n):
    for j in range(n):
        R_pi[i*n+j] = 0
        # Check State A and B first since we can move out of bounds from these states, but we always end up at A1 or B1 from A
        # and B
        if (i, j) == A: 
            R_pi[i*n+j] = 10.0
        elif (i, j) == B:
            R_pi[i*n+j] = 5.0
        else:
            R_up = 0
            R_down = 0
            R_left = 0
            R_right = 0
            if i - 1 < 0:
                R_up = -1
            if i + 1 >= n:
                R_down = -1
            if j + 1 >= n:
                R_right = -1
            if j - 1 < 0:
                R_left = -1
            R_pi[i*n+j] = 0.25 * (R_up + R_down + R_right + R_left)

# Set up the transition matrix
# For A and B, probability that the agent will move from A to A1 and B to B1 will be 1 since all actions move the agent to A1
# B1

# For positions on the border of the grid, the agent may stay in the same position depending on the number of sides of the square
# that are common with the boundary (this is stored in border count)

for i in range(n):
    for j in range(n):
        up = i-1
        down = i+1
        right = j+1
        left = j-1
        bordercount = 0
        if (i, j) == A:
            P_pi[i*n+j][A1[0]*n+A1[1]] = 1
        elif (i, j) == B:
            P_pi[i*n+j][B1[0]*n+B1[1]] = 1
        else:
            if up < 0:
                bordercount += 1
            else:
                P_pi[i*n+j][(i-1)*n+j] = 0.25
            if left < 0:
                bordercount += 1
            else:
                P_pi[i*n+j][i*n+(j-1)] = 0.25
            if right >= n:
                bordercount += 1
            else:
                P_pi[i*n+j][i*n+(j+1)] = 0.25
            if down >= n:
                bordercount += 1
            else:
                P_pi[i*n+j][(i+1)*n+j] = 0.25

        P_pi[i*n+j][i*n+j] = 0.25 * bordercount
            
V_pi = np.matmul(np.linalg.inv(identity - gamma*P_pi), R_pi)


# In[4]:


print (V_pi.reshape((n,n)))


# In[5]:


"""
* Iterative Policy Evaluation. Alternate Method
"""

grid = np.zeros((n, n))

delta = 1e-4
diff = 1000
while diff > delta:
    updated_grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
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
                updated_grid[i, j] += action_prob * (reward + gamma * (grid[x, y]))
    diff = np.abs(np.sum(np.subtract(updated_grid, grid)))
    grid = updated_grid


# In[6]:


print (grid)

