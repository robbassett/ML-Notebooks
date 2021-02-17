

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


import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

def get_neighbours(x, y):
    result = []
    for i in (-1, +1):
        result.append(((x + i + 7) % 7, y))
        result.append((x, (y + i + 11) % 11))
    return result


def agent(obs_dict, config_dict, gindex):
    #################################################
    # State retrieval
    #################################################
    #print("-----------")
    global last_action
    conf = Configuration(config_dict)
    obs = Observation(obs_dict)
    step = obs.step + 1
    my_idx = gindex
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


class Agent:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation, gindex):
        return agent(observation, self.configuration, gindex)
        
