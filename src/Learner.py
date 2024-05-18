from abc import ABC, abstractmethod
from typing import List, Dict

from Action import Action
from QTable import QTable
from State import State


class Learner(ABC):
    @abstractmethod
    def __init__(self, q_table: QTable, params: Dict[str, float]) -> None:
        self.q_table = q_table
        self.params = params

    @abstractmethod
    def learn(self, possible_actions: List[Action], state: State, action: Action,
              next_state: State, reward: int, done: bool) -> None:
        """
        The learn method updates the q-values in the q-table based on the reward received
        :param possible_actions: All the possible actions in the current state.
        :param state: The current state.
        :param action: The performed action.
        :param next_state: The next state.
        :param reward: The reward to be received.
        :param done: If the episode is done.
        """
        pass


class QLearning(Learner):

    def __init__(self, q_table: QTable, params: Dict[str, float]) -> None:
        super().__init__(q_table, params)

    def learn(self, possible_actions: List[Action], state: State, action: Action,
              next_state: State, reward: int, done: bool) -> None:

        if done:
            return

        lr = self.params.get('lr', 0.7)
        gamma = self.params.get('gamma', 0.9)

        old_q = self.q_table.get_q(state, action)

        next_action = max(possible_actions, key=lambda a: self.q_table.get_q(next_state, a))

        next_q = self.q_table.get_q(next_state, next_action)
        new_q = old_q + lr * (reward + gamma * next_q - old_q)
        self.q_table.set_q(state, action, new_q)


class SARSA(Learner):

    def __init__(self, q_table: QTable, params: Dict[str, float]) -> None:
        super().__init__(q_table, params)

    def learn(self, state: State, action: Action, next_state: State, next_action: Action,
              reward: float, done: bool) -> None:
        if done:
            return

        lr = self.params.get('lr', 0.7)
        gamma = self.params.get('gamma', 0.9)

        old_q = self.q_table.get_q(state, action)
        next_q = self.q_table.get_q(next_state, next_action)
        new_q = old_q + lr * (reward + gamma * next_q - old_q)
        self.q_table.set_q(state, action, new_q)
