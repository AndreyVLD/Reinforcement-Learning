from copy import deepcopy
from typing import Tuple, List, Any

from Maze import Maze
from Agent import Agent
from Action import Action
from QTable import QTable
from ExplorationStrategy import ExplorationStrategy
from tqdm import tqdm
from Learner import Learner, QLearning, SARSA
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np


def choose_action(exploration_strategy: ExplorationStrategy, function: str, agent: Agent, maze: Maze,
                  temperature: float, epsilon: float) -> Action:
    """
    Choose an action based on the exploration strategy.
    :param exploration_strategy: The exploration object that keeps track of what choices to make at each step.
    :param function: The exploration strategy to use.
    :param agent: The agent that is making the choice.
    :param maze: The maze in which the agent is moving.
    :param temperature: The temperature parameter for the Boltzmann exploration strategy.
    :param epsilon: The epsilon parameter for the e-greedy exploration strategy.
    :return: The next action to take.
    """
    if function == 'boltzmann':
        action = exploration_strategy.boltzmann(agent, maze, temperature)
    elif function == 'uniform':
        action = exploration_strategy.random(agent, maze)
    elif function == 'greedy':
        action = exploration_strategy.e_greedy(agent, maze, epsilon)
    else:
        raise Exception()
    return action


def plot_distances(all_episodes_lengths: list) -> None:
    """
    Plot the average number of steps per episode.
    :param all_episodes_lengths: The lengths of all episodes.
    :return: None, it plots the graph
    """
    mlp.use('TkAgg')
    arr = np.array(all_episodes_lengths)
    arr_mean = np.mean(arr, axis=0)
    plt.plot(np.arange(1, arr_mean.size + 1), arr_mean)
    plt.xlabel('Episode')
    plt.ylabel('Average Number of Steps')
    plt.title('Average Number of Steps per Episode - Easy Maze')
    plt.show()


def run_agent(learner: Learner, maze: Maze, agent: Agent, n_episodes: int, exploration_strategy: ExplorationStrategy,
              function: str, temperature: float = 0.5, epsilon: float = 0.3) \
        -> tuple[list[Any], list[tuple[Any, Any]]]:
    """
    The main loop for running the agent in the maze.
    :param learner: The learner object that keeps track of the Q-values.
    :param maze: The maze in which the agent is moving.
    :param agent: The agent that is moving in the maze.
    :param n_episodes: The number of episodes to run the agent for.
    :param exploration_strategy: The exploration strategy object that will keep track of the next action choice.
    :param function: The exploration strategy to use.
    :param temperature: The temperature parameter for the Boltzmann exploration strategy.
    :param epsilon: The epsilon parameter for the e-greedy exploration strategy.
    :return: The lengths of all episodes and the steps taken in the final episode, used for visualization.
    """
    episode_lengths = []
    steps_final_episode = []

    for _ in tqdm(range(n_episodes)):

        agent.reset()
        steps_final_episode = []
        state = agent.get_state(maze)
        action = choose_action(exploration_strategy, function, agent, maze, temperature, epsilon)
        done = False

        while not done:

            steps_final_episode.append((state.y, state.x))
            next_state, reward, done = agent.step(action, maze)
            possible_actions = agent.get_valid_actions(maze)

            next_action = choose_action(exploration_strategy, function, agent, maze, temperature,
                                        epsilon)

            if isinstance(learner, QLearning):
                learner.learn(possible_actions, state, action, next_state, reward, done)
            elif isinstance(learner, SARSA):
                learner.learn(state, action, next_state, next_action, reward, done)

            state = next_state
            action = next_action
        episode_lengths.append(agent.nr_of_actions_since_reset)

        steps_final_episode.append((state.y, state.x))

    return episode_lengths, steps_final_episode


def main() -> None:
    """
    The main function to run the agent in the maze.
    """
    maze = Maze("./../data/easy_maze.txt")

    # Locations of the rewards and end of the maze
    maze.set_reward(x=24, y=14, reward=10)
    maze.set_terminal(x=24, y=14)

    # Create an Agent.
    agent = Agent(start_x=0, start_y=0)

    # Create a QTable.
    states = maze.get_all_states()
    actions = [Action(i) for i in ["up", "down", "left", "right"]]
    q_table = QTable(states, actions, 10, 0)

    # Create an ExplorationStrategy.
    exploration_strategy = ExplorationStrategy(q_table)

    # Create a learner.
    params = {"lr": 0.7, "gamma": 0.9}
    learner = QLearning(q_table, params)

    n_runs = 10
    episodes = 300

    all_episode_lengths = []
    steps = []
    copy_arr = deepcopy(q_table)

    # Training n different runs to evaluate average performance
    for _ in range(n_runs):
        episode_lengths, steps = run_agent(learner, maze, agent, episodes, exploration_strategy, 'greedy',
                                           0.3, 0.1)
        all_episode_lengths.append(episode_lengths)

        new_table = deepcopy(copy_arr)
        exploration_strategy = ExplorationStrategy(new_table)
        learner = QLearning(new_table, params)

    # Plotting the trajectory of the agent
    plot_distances(all_episode_lengths)
    maze.visualize([])
    maze.visualize(steps)


if __name__ == "__main__":
    main()
