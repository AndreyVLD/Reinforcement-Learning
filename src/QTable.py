from Action import Action
from State import State
from typing import List

"""
A q-table stores the q-values for each possible state-action pair.
"""


class QTable:
    def __init__(self, states: List[State], actions: List[Action], r_max: float, y: float):
        """
        This is a pessimistic initialization. Try an optimistic initialization instead: RMax / 1 - y.
        """
        initial_value = 0
        self.q_table = {s.id: {a.id: r_max / (1 - y) for a in actions} for s in states}

    def get_q(self, state: State, action: Action):
        return self.q_table[state.id][action.id]

    def set_q(self, state: State, action: Action, value: float):
        self.q_table[state.id][action.id] = value
