from QTable import QTable
from Agent import Agent
from Maze import Maze
import numpy as np
import random


class ExplorationStrategy:

    def __init__(self, q_table: QTable):
        self.q_table = q_table

    def random(self, agent: Agent, maze: Maze):
        """
        The random exploration strategy selects a random action uniformly at random
        from the set of all valid actions.
        """
        valid_actions = agent.get_valid_actions(maze)
        next_action = np.random.randint(0, len(valid_actions))
        return valid_actions[next_action]

    def e_greedy(self, agent: Agent, maze: Maze, eps: float):
        """
        The e-greedy exploration strategy selects a random action with probability eps,
        and the action with highest q-value with probability 1 - eps. A value of epsilon
        close to 0 favours exploitation, while a value close to 1 favours exploration.
        """

        valid_actions = agent.get_valid_actions(maze)

        prob = np.random.uniform(0, 1)
        action = None

        if prob < eps:
            next_action = np.random.randint(0, len(valid_actions))
            action = valid_actions[next_action]
        else:
            max_q = -np.inf
            state = agent.get_state(maze)
            for a in valid_actions:
                q = self.q_table.get_q(state, a)
                if max_q < q:
                    action = a
                    max_q = q

        return action

    def boltzmann(self, agent: Agent, maze: Maze, temperature: float):
        """
        The Boltzmann exploration strategy assigns a probability to each action based on its estimated q-values.
        A large value of the temperature encourages exploration, and as the temperature declines over time,
        exploitation is favoured. 
        """
        valid_actions = agent.get_valid_actions(maze)
        probabilities = []
        state = agent.get_state(maze)
        sums = 0

        for a in valid_actions:
            q = self.q_table.get_q(state, a)
            exp_q = np.exp(q / temperature)
            probabilities.append(exp_q)
            sums += exp_q

        probabilities = [p / sums for p in probabilities]

        next_choice = np.random.choice(valid_actions, p=probabilities)
        return next_choice
