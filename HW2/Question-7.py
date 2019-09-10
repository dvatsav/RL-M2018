
# coding: utf-8

# In[1]:


from copy import deepcopy
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic('matplotlib inline')


# In[2]:


"""
* Without modifications, with max_cars = 10 due to runtime constraints
"""
credit = 10
move_cost = 2
mean_req_first = 3
mean_req_second = 4
mean_ret_first = 3
mean_ret_second = 2
max_cars = 10
max_moves = 5
discount_factor = 0.9


# In[3]:


values = np.zeros((max_cars+1, max_cars+1)) # Since we can have 0 to num_cars
policy = np.zeros((max_cars+1, max_cars+1), dtype=int)
actions = np.arange(-max_moves, max_moves+1)


# In[4]:


# Precompute poisson PMFs because it takes a long time otherwise
poisson_pmfs = {}
for i in range(max_cars*3):
    poisson_pmfs[(i, mean_req_first)] = poisson.pmf(i, mean_req_first)
    poisson_pmfs[(i, mean_req_second)] = poisson.pmf(i, mean_req_second)
    poisson_pmfs[(i, mean_ret_first)] = poisson.pmf(i, mean_ret_first)
    poisson_pmfs[(i, mean_ret_second)] = poisson.pmf(i, mean_ret_second)


# In[5]:


def get_return(i, j, action, values):
    return_val = 0.0
    
    # Discard the extra cars and account for those that have moved
    remaining_cars_first = min(i - action, max_cars)
    remaining_cars_second = min(j + action, max_cars)
    
    # Cost of moving cars according to the selected action
    return_val -= move_cost * np.abs(action)
    
    # req_1 is the number of loan requirements at location 1
    # req_2 is the number of loan requirements at location 2
    for req_1 in range(max_cars+1):
        for req_2 in range(max_cars+1):
            for ret_1 in range(max_cars+1):
                for ret_2 in range(max_cars+1):
                    num_cars_first = remaining_cars_first
                    num_cars_second = remaining_cars_second
                    
                    rented_out_first = min(num_cars_first, req_1)
                    rented_out_second = min(num_cars_second, req_2)

                    reward = (rented_out_first + rented_out_second) * credit

                    num_cars_first -= rented_out_first
                    num_cars_second -= rented_out_second

                    num_cars_first_ = min(num_cars_first + ret_1, max_cars)
                    num_cars_second_ = min(num_cars_second + ret_2, max_cars)
                    # print (num_cars_first_, num_cars_second_)
                    return_val += (poisson_pmfs[(req_1, mean_req_first)] *                                    poisson_pmfs[(req_2, mean_req_second)] *                                    poisson_pmfs[(ret_1, mean_ret_first)] *                                    poisson_pmfs[(ret_2, mean_ret_second)]) *                     (reward + discount_factor * values[num_cars_first_, num_cars_second_])

    return return_val

def evaluate_policy():
    
    theta = 1e-4
    while True:
        delta = 0
        for i in range(max_cars+1):
            for j in range(max_cars+1):
                old_value_s = values[i, j]
                action = policy[i, j]
                values[i, j] = get_return(i, j, action, values)
                delta = max(delta, np.abs(old_value_s - values[i, j]))
        print ("Delta:", delta)
        if delta < theta:
            break
    return values
        
def improve_policy():
    stable = True
    for i in range(max_cars+1):
        for j in range(max_cars+1):
            old_policy = policy[i, j]
            returns = []
            for action in actions:
                if ((0 <= action <= i) or (-j <= action <= 0)):
                    returns.append(get_return(i, j, action, values))
                else:
                    returns.append(-np.inf)
            policy[i, j] = actions[returns.index(max(returns))]
            if policy[i, j] != old_policy:
                print ("Policy Change for State:", (i, j), "New Policy:", policy[i, j], "Old Policy:", old_policy)
                stable = False
    return stable


# In[6]:


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
    print (policy)
    print ("-----------------------------")
    if stable:
       break 


# In[7]:


policy


# In[8]:


values


# In[9]:


plt.imshow(policy, origin='lower')
plt.colorbar()
plt.xlabel("# cars at the second location")
plt.ylabel("# cars at the first location")
plt.show()


# In[10]:


"""
* With modifications, with max_cars = 10 due to runtime constraints
"""

credit = 10
move_cost = 2
mean_req_first = 3
mean_req_second = 4
mean_ret_first = 3
mean_ret_second = 2
max_cars = 10
max_moves = 5
discount_factor = 0.9
free_shuttles = 1
parking_limit = 5 # set to a value less than the max cars per location
parking_cost = 4


# In[11]:


values = np.zeros((max_cars+1, max_cars+1)) # Since we can have 0 to num_cars
policy = np.zeros((max_cars+1, max_cars+1), dtype=int)
actions = np.arange(-max_moves, max_moves+1)


# In[12]:


# Precompute poisson PMFs because it takes a long time otherwise
poisson_pmfs = {}
for i in range(max_cars*3):
    poisson_pmfs[(i, mean_req_first)] = poisson.pmf(i, mean_req_first)
    poisson_pmfs[(i, mean_req_second)] = poisson.pmf(i, mean_req_second)
    poisson_pmfs[(i, mean_ret_first)] = poisson.pmf(i, mean_ret_first)
    poisson_pmfs[(i, mean_ret_second)] = poisson.pmf(i, mean_ret_second)     


# In[13]:


def get_return(i, j, action, values):
    return_val = 0.0
    
    # Check for free shuttles and reduce reward due to moving cars
    if action > 0:
        return_val -= move_cost * (np.abs(action) - free_shuttles)
    else:
        return_val -= move_cost * np.abs(action)
    
    # Discard the extra cars and account for those that have moved overnight
    remaining_cars_first = min(i - action, max_cars)
    remaining_cars_second = min(j + action, max_cars)
    
    # Check overnight parking deductions after moving of cars
    parking_deduction = 0
    if remaining_cars_first > parking_limit:
        return_val -= parking_cost
    if remaining_cars_second > parking_limit:
        return_val -= parking_cost
    
    # req_1 is the number of loan requirements at location 1
    # req_2 is the number of loan requirements at location 2
    for req_1 in range(max_cars+1):
        for req_2 in range(max_cars+1):
            for ret_1 in range(max_cars+1):
                for ret_2 in range(max_cars+1):
                    num_cars_first = remaining_cars_first
                    num_cars_second = remaining_cars_second
                    
                    rented_out_first = min(num_cars_first, req_1)
                    rented_out_second = min(num_cars_second, req_2)

                    reward = (rented_out_first + rented_out_second) * credit

                    num_cars_first -= rented_out_first
                    num_cars_second -= rented_out_second

                    num_cars_first_ = min(num_cars_first + ret_1, max_cars)
                    num_cars_second_ = min(num_cars_second + ret_2, max_cars)
                    # print (num_cars_first_, num_cars_second_)
                    return_val += (poisson_pmfs[(req_1, mean_req_first)] *                                    poisson_pmfs[(req_2, mean_req_second)] *                                    poisson_pmfs[(ret_1, mean_ret_first)] *                                    poisson_pmfs[(ret_2, mean_ret_second)]) *                     (reward + discount_factor * values[num_cars_first_, num_cars_second_])

    return return_val

def evaluate_policy():
    
    theta = 1e-4
    while True:
        delta = 0
        for i in range(max_cars+1):
            for j in range(max_cars+1):
                old_value_s = values[i, j]
                action = policy[i, j]
                values[i, j] = get_return(i, j, action, values)
                delta = max(delta, np.abs(old_value_s - values[i, j]))
        print ("Delta:", delta)
        if delta < theta:
            break
    return values
        
def improve_policy():
    stable = True
    for i in range(max_cars+1):
        for j in range(max_cars+1):
            old_policy = policy[i, j]
            returns = []
            for action in actions:
                if ((0 <= action <= i) or (-j <= action <= 0)):
                    returns.append(get_return(i, j, action, values))
                else:
                    returns.append(-np.inf)
            policy[i, j] = actions[returns.index(max(returns))]
            if policy[i, j] != old_policy:
                print ("Policy Change for State:", (i, j), "New Policy:", policy[i, j], "Old Policy:", old_policy)
                stable = False
    return stable


# In[14]:


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
    print (policy)
    print ("-----------------------------")
    if stable:
       break 


# In[15]:


policy


# In[16]:


values


# In[17]:


plt.imshow(policy, origin='lower')
plt.colorbar()
plt.xlabel("# cars at the second location")
plt.ylabel("# cars at the first location")
plt.show()

