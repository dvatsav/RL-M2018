"""
* Comparison between UCB, optimistic value and e-greedy algorithms
* k = 10
"""

import numpy as np
from random import random as rand, randint as randrange
import argparse
import matplotlib.pyplot as plt
import math


class KArmTestBed:
    def __init__(self, num_simulations, time_steps, k):
        self.num_simulations = num_simulations
        self.time_steps = time_steps
        self.k = k
        self.results = {}

    """
    * Add a test
    """

    def add_test(self, test):
        self.test = test
        self.epsilon = 0
        if self.test == "e-greedy":
            self.epsilon = 0.1
        elif self.test == "optimistic":
            self.epsilon = 0
        self.results[self.test] = 0

    """
    * Run a test
    * Stores average reward and average cumulative reward at time t for the test
    * qa starts off with equal values, and take random walks
    """

    def run_test(self):
        total_reward_at_time_t = [0] * self.time_steps
        total_cumulative_reward_at_time_t = [0] * self.time_steps
        total_optimal_actions_at_time_t = [0] * self.time_steps
        for sim in range(self.num_simulations):
            if args.env == 's':
                qa = np.random.normal(0, 1, self.k)
                optimal_action_arm = np.argmax(qa)
            else:
                qa = [0] * self.k
                optimal_action_arm = 0

            # No epsilon greedy, but with optimistic estimate vs epsilon greedy with optimistic estimate
            if self.test == "optimistic":
                action_reward_estimate = [5] * self.k
            else:
                action_reward_estimate = [0] * self.k
            cumulative_reward = 0
            o = 0
            num_pulls = [0] * self.k
            for time in range(self.time_steps):

                # If env is non stationary, modify true values at each time step
                if args.env != 's':
                    qa = list(map(lambda x: x + np.random.normal(0, 0.01), qa))
                    optimal_action_arm = qa.index(max(qa))

                if self.test != "UCB":
                    if rand() > self.epsilon:
                        arm = action_reward_estimate.index(max(action_reward_estimate))
                    else:
                        arm = randrange(0, self.k-1)
                else:
                    mod_num_pulls = [i for i in num_pulls]
                    mod_num_pulls = [0.001 if x == 0 else x for x in mod_num_pulls]
                    l = [(action_reward_estimate[i] + args.confidence *
                         math.sqrt(math.log(time+1)/mod_num_pulls[i])) for i in range(self.k)] 
                    arm = l.index(max(l))


                if arm == optimal_action_arm:
                    total_optimal_actions_at_time_t[time] += 1

                reward = np.random.normal(qa[arm], 1)
                total_reward_at_time_t[time] += reward

                cumulative_reward += reward
                total_cumulative_reward_at_time_t[time] += cumulative_reward

                num_pulls[arm] += 1
                if args.env == 'n':
                    alpha = args.alpha
                    o = o + alpha*(1-o)
                    beta = alpha/o

                    action_reward_estimate[arm] = action_reward_estimate[arm] + \
                        beta * (reward - action_reward_estimate[arm])
                else:
                    
                    alpha = (1/(num_pulls[arm]))
                    action_reward_estimate[arm] = action_reward_estimate[arm] + \
                        alpha * (reward - action_reward_estimate[arm])

        self.results[self.test] = {
            "Average Cumulative Reward at time t": list(map(lambda x: x / self.num_simulations, total_cumulative_reward_at_time_t)),
            "Average Reward at time t": list(map(lambda x: x / self.num_simulations, total_reward_at_time_t)),
            "Percentage Optimal Action at time t": list(map(lambda x: (x / self.num_simulations) * 100, total_optimal_actions_at_time_t))
        }

    """
    * Plot the average reward at each time step.
    * Plot the percentage of optimal action at each time step
    """

    def plot_results(self):
        legend = []
        plt.figure(1)
        for test in self.results:
            plt.plot(np.arange(self.time_steps),
                     self.results[test]["Average Reward at time t"])
            legend.append(test)
        plt.legend(legend, loc='lower right')
        plt.xlabel("Time Steps")
        plt.ylabel("Average Reward")

        plt.figure(2)
        for test in self.results:
            plt.plot(np.arange(self.time_steps),
                     self.results[test]["Percentage Optimal Action at time t"])
            legend.append(test)
        plt.legend(legend, loc='lower right')
        plt.xlabel("Time Steps")
        plt.ylabel("% Optimal Action")
        plt.show()


if __name__ == '__main__':

    """
    * Validate timesteps and number of simulations
    """
    def check_positive(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid positive int value" % value)
        return ivalue

    """
    * Validate the value of alpha entered by the user
    """
    def check_alpha(value):
        epsilon_alpha = float(value)
        if epsilon_alpha < 0 or epsilon_alpha > 1:
            raise argparse.ArgumentTypeError(
                "%s is an invalid epsilon/alpha value" % str(value))
        return epsilon_alpha

    """
    * Parse Arguments
    """
    parser = argparse.ArgumentParser(
        description='This program runs a K-Arm Test Bed simulation')
    parser.add_argument('-s', '--num_simulations', action='store',
                        help="Number of Simulations to Run", type=check_positive, default=2000)
    parser.add_argument('-t', '--time_steps', action='store',
                        help="Number of Time Steps per simulation", type=check_positive, default=1000)
    parser.add_argument('-k', '--num_arms', action='store',
                        help="Number of Arms", type=check_positive, default=10)
    parser.add_argument('-a', '--alpha', default=0.1, action='store',
                        help="Constant step size for Value update", type=check_alpha)
    parser.add_argument('-c', '--confidence', default=2,
                        help="Constant for UCB selection", type=check_positive)
    parser.add_argument('--env', action='store',
                        help='Stationary or Non Stationary env. -> s/n', default='s')
    args = parser.parse_args()

    """
    * Initialize and run simulations, plot results
    """
    bandit = KArmTestBed(args.num_simulations, args.time_steps, args.num_arms)
    print("Test Conditions:\nNumber of Simulations per Test: {}\nNumber of Time Steps per Simulation: {}\nNumber of Arms: {}\n"
          .format(args.num_simulations, args.time_steps, args.num_arms))

    tests = ["e-greedy", "optimistic", "UCB"]
    for test in range(3):
        print("Running test {} - {}".format(
            test+1, tests[test]))
        bandit.add_test(tests[test])
        bandit.run_test()
    bandit.plot_results()
