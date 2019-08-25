"""
* Exercise 2.6, Sutton.
* Generalized Bandit Prolem - For values specified in the exercise, use - 
* k = 10
"""

import numpy as np
from random import random as rand, randint as randrange
import argparse
import matplotlib.pyplot as plt


class KArmTestBed:
    def __init__(self, num_simulations, time_steps, k):
        self.num_simulations = num_simulations
        self.time_steps = time_steps
        self.k = k
        self.results = {}

    """
    * Add a test with a particular epsilon. Each test case is associated with one epsilon
    * value.
    """

    def add_test(self, epsilon):
        self.epsilon = epsilon
        self.results[self.epsilon] = 0

    """
    * Run a test with a particular epsilon
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
                
            if self.epsilon == 0:
                action_reward_estimate = [5] * self.k
            else:
                action_reward_estimate = [0] * self.k
            num_pulls = [0] * self.k
            cumulative_reward = 0
            for time in range(self.time_steps):
                
                if args.env != 's':
                    qa = list(map(lambda x: x + np.random.normal(0, 0.01), qa))
                    optimal_action_arm = qa.index(max(qa))

                if rand() > self.epsilon:
                    arm = action_reward_estimate.index(max(action_reward_estimate))
                else:
                    arm = randrange(0, self.k-1)

                if arm == optimal_action_arm:
                    total_optimal_actions_at_time_t[time] += 1

                reward = np.random.normal(qa[arm], 1)
                total_reward_at_time_t[time] += reward

                cumulative_reward += reward
                total_cumulative_reward_at_time_t[time] += cumulative_reward

                num_pulls[arm] += 1
                if args.step == "SampleMean":
                    alpha = (1/(num_pulls[arm]))
                elif args.step == "Constant":
                    alpha = args.alpha

                action_reward_estimate[arm] = action_reward_estimate[arm] + \
                    alpha * (reward - action_reward_estimate[arm])

        self.results[self.epsilon] = {
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
        for epsilon in self.results:
            plt.plot(np.arange(self.time_steps),
                     self.results[epsilon]["Average Reward at time t"])
            legend.append("Epsilon = " + str(epsilon))
        plt.legend(legend, loc='lower right')
        plt.xlabel("Time Steps")
        plt.ylabel("Average Reward")

        plt.figure(2)
        for epsilon in self.results:
            plt.plot(np.arange(self.time_steps),
                     self.results[epsilon]["Percentage Optimal Action at time t"])
            legend.append("Epsilon = " + str(epsilon))
        plt.legend(legend, loc='lower right')
        plt.xlabel("Time Steps")
        plt.ylabel("% Optimal Action")
        plt.show()


"""
* The number of tests and the provided epsilon values should be of same length
"""
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
    * Make sure that the user only uses Sample Mean or Constant step size
    """
    def check_step(value):
        if value not in ["SampleMean", "Constant"]:
            raise argparse.ArgumentTypeError(
                "%s is an invalid step value" % str(value))
        return value

    """
    * Validate the value of epsilon or alpha entered by the user
    """
    def check_epsilon_alpha(value):
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
    parser.add_argument('-n', '--num_tests', action='store',
                        help='Number of tests to run', default=2, type=check_positive)
    parser.add_argument('-e', '--epsilon', nargs='+', default=[0, 0.1],
                        help="Epsilon value for e-greedy algorithm", type=check_epsilon_alpha)
    parser.add_argument('--step', action='store',
                        help="Can take value - SampleMean, Constant", type=check_step, default="Constant")
    parser.add_argument('-a', '--alpha', default=0.1,
                        help="Constant step size for Value update", type=check_epsilon_alpha)
    parser.add_argument('--env', action='store', help='Stationary or Non Stationary env. -> s/n', default='s')
    args = parser.parse_args()

    """
    * Initialize and run simulations, plot results
    """
    bandit = KArmTestBed(args.num_simulations, args.time_steps, args.num_arms)
    print("Test Conditions:\nNumber of Simulations per Test: {}\nNumber of Time Steps per Simulation: {}\nNumber of Arms: {}\n"
          .format(args.num_simulations, args.time_steps, args.num_arms))
    if args.num_tests != len(args.epsilon):
        raise argparse.ArgumentTypeError(
            "Number of tests should be equal to number of epsilons")
    for test in range(args.num_tests):
        print("Running test {} with epsilon {}".format(
            test+1, args.epsilon[test]))
        bandit.add_test(args.epsilon[test])
        bandit.run_test()
    bandit.plot_results()
