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
from models import LibaAgy_mk0 as brain
from models import GooseGym

RoboGoose = GooseLee(0.001,brain,epsilon=0.5,name='GooseLee2D0.1')
Dojo = GooseGym(RoboGoose,ncheck=1,ndisplay=1,border=1,load_weights='models/GooseLee2D0.0.h5')
Dojo.train(250)
