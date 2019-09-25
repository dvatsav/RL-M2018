
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from matplotlib import cm
get_ipython().magic('matplotlib inline')


# In[2]:


actions = {'hit' : 1, 'stick' : 0}
rewards = {'win' : 1, 'lose' : -1, 'draw' : 0}
policy_player = np.ones(22, dtype=int)
policy_player[20] = actions['stick']
policy_player[21] = actions['stick']

policy_dealer = np.ones(22, dtype=int)
for i in range(17, 22, 1):
    policy_dealer[i] = actions['stick']


# In[3]:


def get_card():
    card = randint(2, 14)
    if card == 14:
        return 11
    else:
        return min(10, card)
    
def init_state():
    dealer_card_1 = get_card()
    usable_ace = bool(randint(0, 1))
    player_sum = randint(2, 21)
    action = randint(0, 1)
    dealer_card_2 = get_card()
    
    return (player_sum, dealer_card_1, dealer_card_2, usable_ace, action)

# for the first episode use the initial policy and use improved policy subsequently
def get_policy(player_sum, dealer_card_1, player_usable_ace, state_action_values):
    if np.array_equal(state_action_values, np.zeros((20, 10, 2, 2))):
        return policy_player[player_sum]
    if dealer_card_1 == 11:
        dealer_card_1 = 1
    action_values = state_action_values[player_sum-2, dealer_card_1-1, int(player_usable_ace), :]
    return np.argmax(action_values)


# In[4]:


def generate_episode(state_action_values):
    
    player_sum, dealer_card_1, dealer_card_2, player_usable_ace, action = init_state()
    
    dealer_sum = dealer_card_1 + dealer_card_2
    dealer_usable_ace = int(dealer_card_1 == 11) + int(dealer_card_2 == 11)
    
    if dealer_sum > 21:
        dealer_sum -= 10
        dealer_usable_ace -= 1
    
    player_episode = []
    
    player_episode.append([player_sum, dealer_card_1, player_usable_ace > 0, action])
    
    if player_sum == 21:
        if dealer_sum == 21:
            return player_episode, 0
        else:
            return player_episode, 1
    
    while True:
        
        if action == actions['stick']:
            break
            
        card = get_card()
        if card == 11:
            player_usable_ace += 1
        player_sum += card
        
        # User may have multiple aces due to infinite deck
        while player_sum > 21 and player_usable_ace:
            player_sum -= 10
            player_usable_ace -= 1
        
        if player_sum > 21:
            return player_episode, -1
        
        
        action = get_policy(player_sum, dealer_card_1, player_usable_ace > 0, state_action_values)
        player_episode.append([player_sum, dealer_card_1, player_usable_ace > 0, action])
        
    while True:
        action = policy_dealer[dealer_sum]
        if action == actions['stick']:
            break
            
        card = get_card()
        if card == 11:
            dealer_usable_ace += 1
        dealer_sum += card
        
        while dealer_sum > 21 and dealer_usable_ace:
            dealer_sum -= 10
            dealer_usable_ace -= 1
        
        if dealer_sum > 21:
            return player_episode, 1
        
    if player_sum > dealer_sum:
        return player_episode, 1
    elif player_sum == dealer_sum:
        return player_episode, 0
    else:
        return player_episode, -1
        


# In[7]:


def monte_carlo(num_episodes, first_visit=False):
    
    state_action_values = np.zeros((20, 10, 2, 2))
    state_action_count = np.zeros((20, 10, 2, 2))
    
    for i in range(1, num_episodes+1):
        
        start_state = init_state()
        start_action = randint(0, 1)
        player_episode, reward = generate_episode(state_action_values)
        
        for (player_sum, dealer_card, usable_ace, action) in player_episode:
            first_visit_check = np.zeros((20, 10, 2, 2))
            usable_ace = int(usable_ace)
            if dealer_card == 11:
                dealer_card = 1
            if first_visit and first_visit_check[player_sum-2, dealer_card-1, usable_ace, action] == 1:
                continue
            first_visit_check[player_sum-2, dealer_card-1, usable_ace, action] = 1

            state_action_count[player_sum-2, dealer_card-1, usable_ace, action] += 1
            state_action_values[player_sum-2, dealer_card-1, usable_ace, action] =                 state_action_values[player_sum-2, dealer_card-1, usable_ace, action] +                 (1/state_action_count[player_sum-2, dealer_card-1, usable_ace, action]) *                 (reward - state_action_values[player_sum-2, dealer_card-1, usable_ace, action])

    return state_action_values    

# Plotting referenced from https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html

def plot_5_2():
    state_action_values = monte_carlo(500000, first_visit=True)
    
    states_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    states_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)
    
    cplots = [states_usable_ace, states_no_usable_ace, action_usable_ace, action_no_usable_ace]
    titles = ['Usable Ace. Optimal Values', 'No Usable Ace. Optimal Values', 'Usable Ace. Optimal Policy', 'No Usable Ace. Optimal Policy']
    
    for cplot, title in zip(cplots, titles):
        if 'Policy' in title:
            cplot = cplot[9:, :]
            fig = plt.figure()
            fig = sns.heatmap(np.flipud(cplot), cmap="YlGnBu", xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(11, 22))))
            fig.set_ylabel('Player Score')
            fig.set_xlabel('Dealer Card')
            fig.set_title(title)

            plt.show()
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X = np.arange(1, 11)
            Y = np.arange(12, 22)
            X, Y = np.meshgrid(X, Y)
            Z = cplot[10:, :]

            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
            ax.set_zlim(-1, 1)
            ax.set_xlim(1, 10)
            ax.set_ylim(12, 21)
            ax.set_xlabel('Dealer Card')
            ax.set_ylabel('Player Score')
            ax.set_zlabel('State-Value Function')

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(title)
            plt.show()

       


# In[8]:


plot_5_2()

