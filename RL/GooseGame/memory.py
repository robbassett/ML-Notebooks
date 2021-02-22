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

class PrioritisedMemory():
    def __init__(self,maxsize,inputshape,n_actions):
        self.memsize = maxsize
        self.memcount = -1
        self.state_memory = np.zeros((maxsize,*inputshape),dtype=np.float32)
        self.next_state_memory = np.zeros((maxsize,*inputshape),dtype=np.float32)
        self.reward_memory = np.zeros(maxsize,dtype=np.float32)
        self.action_memory = np.zeros(maxsize,dtype=np.int64)
        self.terminal_memory = np.zeros(maxsize,dtype=np.float32)
        self.mem_TD_loss = np.zeros(maxsize,dtype=np.float32)

    def store_memory(self,state, action, reward, terminal,state_,TDE):
        if self.memcount < self.memsize or TDE <= self.mem_TD_loss.min():
            index = self.memcount%self.memsize
            self.state_memory[index] = state
            self.next_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = terminal
            self.mem_TD_loss[index] = TDE
            self.memcount+=1

    def get_batch(self,batch_size):
        TDE_sort = np.argsort(self.mem_TD_loss)[::-1]
        hbs = int(batch_size/2)
        pb = TDE_sort[:hbs]
        rb = np.random.choice(TDE_sort[hbs:],hbs,replace=False)
        batch = np.concatenate((pb,rb))
        
        states = np.array(self.state_memory[batch])
        states_ = np.array(self.next_state_memory[batch])
        rewards = np.array(self.reward_memory[batch])
        actions = np.array(self.action_memory[batch])
        terminals = np.array(self.terminal_memory[batch])

        return states, actions, rewards, terminals, states_
