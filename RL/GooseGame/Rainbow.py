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

from agents import Goose_mk1 as GooseLee
from models import LibaAgy_lite as brain
from models import GooseGym as Gym

from coaches.kagglegreedy import GreedyAgent as Greedy
from coaches.diffusion_agent import Agent as Diffusion
from coaches.crazy_goose import Crazy
from coaches.risk_averse_greedy import RAGreedy
from coaches.straightforward_bfs import BFS
from coaches.RLbot import RLbot

rbots = [Greedy,Diffusion,Crazy,RAGreedy,BFS,RLbot]

RoboGoose = GooseLee(0.001,brain,epsilon=1.0,name='test')
Dojo = Gym(RoboGoose,ncheck=25,n_smart=1,rule_bots=rbots,ndisplay=1000000,switch_epoch=10000)#,load_weights='models/GooseLee_lite2.0.h5')
Dojo.train(50)

F = plt.figure()
ax = F.add_subplot(111)
ax.hist(RoboGoose.M.mem_TD_loss,bins=100)
plt.show()
