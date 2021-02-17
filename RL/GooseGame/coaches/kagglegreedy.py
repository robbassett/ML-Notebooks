import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from random import choice, sample
from typing import *
from enum import auto, Enum

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap as LSC
mycm = LSC.from_list("tmcm1",['k','r','tab:orange','tab:green','dodgerblue'],N=1000)

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, row_col
from kaggle_environments import make

""" BASE GREEDY AGENT (KAGGLE)"""
class Action(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    def to_row_col(self):
        if self == Action.NORTH:
            return -1, 0
        if self == Action.SOUTH:
            return 1, 0
        if self == Action.EAST:
            return 0, 1
        if self == Action.WEST:
            return 0, -1
        return 0, 0

    def opposite(self):
        if self == Action.NORTH:
            return Action.SOUTH
        if self == Action.SOUTH:
            return Action.NORTH
        if self == Action.EAST:
            return Action.WEST
        if self == Action.WEST:
            return Action.EAST
        raise TypeError(str(self) + " is not a valid Action.")

def translate(position: int, direction: Action, columns: int, rows: int) -> int:
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column


def adjacent_positions(position: int, columns: int, rows: int) -> List[int]:
    return [
        translate(position, action, columns, rows)
        for action in Action
    ]


def min_distance(position: int, food: List[int], columns: int):
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )

class GreedyAgent:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation, gindex):
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation['food']
        geese = observation['geese']
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != gindex and len(goose) > 0
        ]

        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        # Move to the closest food
        position = geese[gindex][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])
        self.last_action = action
        return action.name


