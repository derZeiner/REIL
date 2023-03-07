# This script requires
# matplotlib >= 3.5.1
# numpy >= 1.22.2
# math

"""
The aim of this script is to simulate a situation in which an agent repeatedly has to choose
betweeen the same options, of which there are finitely many.
Each option leads to a reward drawn from a fixed Gaussian distribution: a multi-armed bandit with Gaussian rewards.
The aim of the agent is to repeatedly interact with the environment to learn which arm to pull.
To do so the agent maintains numerical estimates for the value of each arm (action-value method).

In this exercise some methods have been removed, you can recognize them by the 'pass' statement.
"""


class GaussianBandit(object):
    """
    This represents a multi-armed bandit whose arms have normally distributed rewards.

    Attributes
    ----------
    means_list : list[float]
        An ordered list of the extected values of the arms.
    stdev_list : list[float]
        An ordered list of the standard deviations of the arms.
    q_values : list[float]
        An ordered list of q-values for the arms.
    action_count : list[int]
        An ordered list whose entries are the number of times an arm has been pulled.
    action_history : list[int]
        A list of the actions that have been taken in the past.
    reward_history : list[float]
        A list of the rewards that have been observed in the past.
    q_history : list[list[float]]
        A list of the historical q-value vectors.
    """

    def __init__(self, means_list, stdev_list=None):
        if stdev_list != None and len(means_list) != len(stdev_list):
            raise ValueError('The list of means and the list of standard deviations are not of equal lengths.')
        # attributes describing the bandit
        self.means_list = means_list
        if stdev_list is None:
            stdev_list = [1] * len(means_list)
        self.stdev_list = stdev_list
        # attributes describing state of knowledge
        self.q_values = [0] * len(means_list)
        self.action_count = [0] * len(means_list)
        # attributes describing a learning history
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(means_list)]

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        """
        Helper function for online averaging.
        """
        # online averaging formula
        return old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)

    def _num_arms(self):
        "Returns the number of arms."
        return len(self.means_list)

    def reset(self, reset_q_values: bool = True):
        """
        Resets all history attributes, and potentially also the q-values and action-counts, of the object in-place.
        """
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(self.means_list)]
        if reset_q_values:
            self.q_values = [0] * len(self.means_list)
            self.action_count = [0] * len(self.means_list)

    def reward(self, arm: int):
        """
        Returns the reward of the respective arm.
        Does not modify the object.
        """
        if arm < 0 or arm > self._num_arms() - 1:
            raise ValueError("This arm does not exist.")
        from numpy.random import normal
        return normal(self.means_list[arm], self.stdev_list[arm])

    def eps_greedy_action(self, epsilon: float):
        """
        Returns the index of an arm chosen according to epsilon-greedy action-choice with respect to the q-values of the object.
        Does not modify the object.
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon is not in the interval [0,1].")

        from numpy.random import random
        if random() < epsilon:
            from numpy.random import randint
            return randint(self._num_arms())
        else:
            return self.q_values.index(max(self.q_values))

    def ucb_action(self, exp_factor: float = 1):
        """
        Returns the index of an arm chosen according to UCB action-choice with respect to the q-values and action-counts of the object.
        Does not modify the object.
        """

        if exp_factor < 0:
            raise ValueError("The exploration factor can not be negative.")

        import math
        ucb_values = []
        for i in range(self._num_arms()):
            if self.action_count[i] == 0:
                ucb_values.append(math.inf)
            else:
                ucb_values.append(
                    self.q_values[i] + exp_factor * math.sqrt(math.log(sum(self.action_count)) / self.action_count[i]))
        return ucb_values.index(max(ucb_values))

    def play(self, action):
        '''
        plays the game for one step with the given action and updates the q-values
        :param action:
        :return: nothing
        '''
        # update history
        self.action_history.append(action)
        # get reward
        reward = self.reward(action)
        # update action-counts
        self.action_count[action] += 1
        # update q-values
        self.q_values[action] = self._update_average(self.q_values[action], self.action_count[action], reward)
        # update history
        self.reward_history.append(reward)
        self.q_history.append(self.q_values.copy())

    def play_eps_greedy(self, num_steps=100, epsilon=0.1):
        '''
        plays the game for num_steps times with epsilon as the probability of choosing a random action
        :param num_steps:
        :param epsilon:
        :return:
        '''
        for i in range(num_steps):
            # choose action with epsilon-greedy policy
            action = self.eps_greedy_action(epsilon)
            self.play(action)

    def play_ucb(self, num_steps=100, exp_factor=1):
        '''
        plays the game for num_steps times with exp_factor as the exploration factor
        :param num_steps:
        :param exp_factor:
        :return:
        '''
        for i in range(num_steps):
            # choose action with UCB policy
            action = self.ucb_action(exp_factor)
            # update history
            self.play(action)

    def plot_action_history(self, title: str = ""):
        """
        Displays a graphical representation of the action-history to standard output.
        The fraction of times an action has been chosen in the past is displayed.
        """
        import matplotlib.pyplot as plt
        # fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Fraction of action in past")
        for i in set(self.action_history):
            actions_chosen = [x == i for x in self.action_history]
            actions_counts = [sum(actions_chosen[0:j + 1]) for j in range(len(actions_chosen))]
            actions_percent = [actions_counts[j] / (j + 1) for j in range(len(actions_counts))]
            plt.plot(actions_percent)
        plt.show()

    def plot_q_history(self, title: str = ""):
        """
        Displays a graphical representation of the q-value-history to standard output.
        """
        import matplotlib.pyplot as plt
        # fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Q-Value")
        plt.plot(self.q_history)
        plt.show()

    def plot_reward_history(self, title=""):
        import matplotlib.pyplot as plt
        # fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plt.scatter(range(len(self.reward_history)), self.reward_history, s=0.3)
        plt.show()


if __name__ == '__main__':
    # set up the bandit for epsilon-greedy
    means = [0.5, 0.6, 0.7]
    stdevs = [0.1, 0.1, 0.1]
    bandit = GaussianBandit(means, stdevs)
    # play the bandit
    bandit.play_eps_greedy(10000, 0.1)
    # plot the results
    bandit.plot_reward_history("Reward History")
    bandit.plot_action_history("Action History")
    bandit.plot_q_history("Q-Value History")

    # set up the bandit for UCB
    means = [0.5, 0.6, 0.7]
    stdevs = [0.1, 0.1, 0.1]
    bandit = GaussianBandit(means, stdevs)
    # play the bandit
    bandit.play_ucb(10000, 1)
    # plot the results
    bandit.plot_reward_history("Reward History")
    bandit.plot_action_history("Action History")
    bandit.plot_q_history("Q-Value History")
