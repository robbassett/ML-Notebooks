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

from models import GooseLee2D_mk0 as GooseLee
from models import LibaAgy_lite as brain
from models import GooseGym

from coaches import kagglegreedy,risk_averse_greedy,diffusion_agent,crazy_goose
from coaches.kagglegreedy import GreedyAgent as Greedy
from coaches.diffusion_agent import Agent as Diffusion

rbots = [Greedy,Diffusion]



RoboGoose = GooseLee(0.0005,brain,epsilon=0.25,name='GooseLee_lite_newr1.0')
Dojo = GooseGym(RoboGoose,nchannel=3,ncheck=10,n_smart=2,rule_bots=rbots,ndisplay=100000,border=1,load_weights='models/GooseLee_lite_newr0.0.h5')
Dojo.train()
