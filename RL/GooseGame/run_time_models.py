import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from random import choice, sample
from typing import *
from enum import auto, Enum

import matplotlib.pyplot as plt

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, row_col
from kaggle_environments import make

# - - - - - - - - - -#
# Rules Based Models #
#- - - - - - - - - - #

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

class GreedyAgent1:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation: Observation):
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
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
        position = geese[observation.index][0]
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

#######################################
############## PARAMETERS #############
#######################################
FOOD_REWARD = 4
BODY_REWARD = -8
POTENTIAL_HEAD_STD_REWARD = -5
PROBABLE_HEAD_FOOD_REWARD = -6
IMPROBABLE_HEAD_FOOD_REWARD = -4
TAIL_REWARD = -6
REVERSE_LAST_REWARD = -12
DIFFUSE_POS_REWARD = 1
DIFFUSE_NEG_REWARD = -1
TAIL_CHASE_REWARD = 8
DIFFUSE_START = 2

DEBUG = False

last_action = None


def get_neighbours(x, y):
    result = []
    for i in (-1, +1):
        result.append(((x + i + 7) % 7, y))
        result.append((x, (y + i + 11) % 11))
    return result


def agent(obs_dict, config_dict):
    #################################################
    # State retrieval
    #################################################
    #print("-----------")
    global last_action
    conf = Configuration(config_dict)
    obs = Observation(obs_dict)
    step = obs.step + 1
    my_idx = obs.index
    my_goose = obs.geese[my_idx]
    my_head = my_goose[0]
    my_row, my_col = row_col(position=my_head, columns=conf.columns)
    if DEBUG:
        print("---------- Step #" + str(step), "- Player #" + str(obs.index))

                  
        
    #################################################
    # Map update
    #################################################
    board = np.zeros((7, 11), dtype=int)

    # Add food to board
    for food in obs.food:
        food_row, food_col = row_col(position=food, columns=conf.columns)
        board[food_row, food_col] += FOOD_REWARD
        '''if DEBUG:
            print("food", food_row, food_col)'''
        
        
    # Iterate over geese to add geese data to board
    nb_geese = len(obs.geese)
    geese_lengths = []
    for i in range(nb_geese):
        '''if DEBUG:
            print("--- Goose #" + str(i))'''
        goose = obs.geese[i]
        potential_food_head = None
        
        # Iterate over cells of current goose
        goose_len = len(goose)
        geese_lengths.append(goose_len)
        '''if DEBUG:
            print("--- Goose #" + str(i) + " len " + str(goose_len))'''
        for j in range(goose_len):
            '''if DEBUG:
                print("--- Goose #" + str(i) + " cell " + str(j))'''
            goose_cell = goose[j]
            goose_row, goose_col = row_col(position=goose_cell, columns=conf.columns)
            
            # Check for food on neighbour cells when handling head
            if j == 0:
                potential_heads = get_neighbours(goose_row, goose_col)                
                for potential_head in potential_heads:
                    for food in obs.food:
                        food_row, food_col = row_col(position=food, columns=conf.columns)
                        if potential_head == (food_row, food_col):
                            potential_food_head = potential_head

            # Update rewards linked to body/tail                  
            if j < goose_len - 1:
                # Body or head
                board[goose_row, goose_col] += BODY_REWARD                
                '''if DEBUG:
                    print("--- Goose #" + str(i) + " cell " + str(j) + " add BODY_REWARD")'''
            else:
                # Tail : may not move if goose eats
                if potential_food_head is not None:
                    board[goose_row, goose_col] += TAIL_REWARD                        
                    '''if DEBUG:
                        print("--- Goose #" + str(i) + " cell " + str(j) + " add TAIL_REWARD")'''
             
        # Update potential villain head positions
        if (i != my_idx) & (goose_len > 0):
            if potential_food_head is not None:
                # Head will prolly go to the food
                for potential_head in potential_heads:
                    if potential_head == potential_food_head:
                        if (board[potential_head[0], potential_head[1]] != BODY_REWARD) & \
                           (board[potential_head[0], potential_head[1]] != TAIL_REWARD):
                            board[potential_head[0], potential_head[1]] += PROBABLE_HEAD_FOOD_REWARD
                            '''if DEBUG:
                                print("--- Goose #" + str(i) + " cell " + str(j) + " add PROBABLE_HEAD_FOOD_REWARD")'''
                    else:
                        if (board[potential_head[0], potential_head[1]] != BODY_REWARD) & \
                           (board[potential_head[0], potential_head[1]] != TAIL_REWARD):
                            board[potential_head[0], potential_head[1]] += IMPROBABLE_HEAD_FOOD_REWARD
                            '''if DEBUG:
                                print("--- Goose #" + str(i) + " cell " + str(j) + " add IMPROBABLE_HEAD_FOOD_REWARD")'''
            else:
                # Standard potential head reward
                for potential_head in potential_heads:
                    if (board[potential_head[0], potential_head[1]] != BODY_REWARD) & \
                       (board[potential_head[0], potential_head[1]] != TAIL_REWARD):
                        board[potential_head[0], potential_head[1]] += POTENTIAL_HEAD_STD_REWARD                                
                        '''if DEBUG:
                            print("--- Goose #" + str(i) + " cell " + str(j) + " add POTENTIAL_HEAD_STD_REWARD")'''
            
            
    # Check if I'm the current longest Goose
    if (len(my_goose) >= max(geese_lengths) - 3) & (step > 8):
        # Chasing my tail as a defensive action makes sense
        my_tail_row, my_tail_col = row_col(position=my_goose[-1], columns=conf.columns)
        board[my_tail_row, my_tail_col] += TAIL_CHASE_REWARD  
        '''if DEBUG:
            print("Adding TAIL_CHASE_REWARD for me")'''
    
    
    # Diffuse values in adjacent cells
    if DEBUG:
        print("Initial board :")
        print(board)
    new_board = board.copy()
    for i in range(7):
        for j in range(11):
            value = board[i, j]
            if value > DIFFUSE_START:
                # Should diffuse positive value
                neighbours = get_neighbours(i, j)                
                for neighbour in neighbours:
                    # Level 1
                    new_board[neighbour] += (2*DIFFUSE_POS_REWARD)
                    
                    # Level 2
                    neighbours_lvl2 = get_neighbours(neighbour[0], neighbour[1])
                    for neighbour_lvl2 in neighbours_lvl2:
                        new_board[neighbour_lvl2] += DIFFUSE_POS_REWARD
            elif value < -DIFFUSE_START:
                # Should diffuse negative value
                neighbours = get_neighbours(i, j)                
                for neighbour in neighbours:
                    # Level 1
                    new_board[neighbour] += (2*DIFFUSE_NEG_REWARD)
                    
                    # Level 2
                    neighbours_lvl2 = get_neighbours(neighbour[0], neighbour[1])
                    for neighbour_lvl2 in neighbours_lvl2:
                        new_board[neighbour_lvl2] += DIFFUSE_NEG_REWARD
    board = new_board  
                        
    
    # Add last_action data to board
    if last_action is not None:
        if last_action == Action.SOUTH.name:
            board[(my_row + 6) % 7, my_col] += REVERSE_LAST_REWARD
        elif last_action == Action.NORTH.name:
            board[(my_row + 8) % 7, my_col] += REVERSE_LAST_REWARD
        elif last_action == Action.EAST.name:
            board[my_row, (my_col + 10)%11] += REVERSE_LAST_REWARD
        elif last_action == Action.WEST.name:
            board[my_row, (my_col + 12)%11] += REVERSE_LAST_REWARD
        '''if DEBUG:
            print("Adding REVERSE_LAST_REWARD for me")'''

    if DEBUG:
        print("Final board :")
        print(board)
                  
            
    #################################################
    # Choose best action
    #################################################
    chosen_action = None
    rewards = []
    potential_next = get_neighbours(my_row, my_col)
    for cell in potential_next:
        rewards.append(board[cell])
    choice = np.argmax(rewards)
    if choice == 0:
        chosen_action = Action.NORTH.name
    elif choice == 1:
        chosen_action = Action.WEST.name
    elif choice == 2:
        chosen_action = Action.SOUTH.name
    else:
        chosen_action = Action.EAST.name
    if DEBUG:
        print("chosen_action", chosen_action)
    last_action = chosen_action
    return chosen_action


class DiffuseAgent:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation):
        return agent(observation, self.configuration)
