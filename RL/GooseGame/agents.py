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

from memory import PrioritisedMemory

class Goose_mk1():

    def __init__(self,lrate,brain,
                     n_actions=4,
                     dimensions=(9,13,3),
                     gamma=0.99,
                     epsilon=1.0,
                     e_dec=2.5e-5,
                     e_min=0.01,
                     retarget=100,
                     memsize=2500,
                     memmin=128,
                     batch_size=128,
                     save_dir='models/',
                     name='GooseLee2D0.1'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.emin = e_min
        self.de = e_dec
        self.steps = 0
        self.n_actions = n_actions
        self.lrate = lrate
        self.dimensions = dimensions
        self.retarget=retarget
        self.memsize=memsize
        self.batch_size=batch_size
        self.N = memmin
        self.actions = np.array(['NORTH', 'SOUTH', 'WEST', 'EAST'])
        self.name = save_dir+name+'.h5'
        self.forbidden = np.array([1,0,3,2])
        self.prev_action = 0
        
        self.M = PrioritisedMemory(self.memsize,self.dimensions,self.n_actions)
        self.Q = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name=name)
        self.V_eval = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name='Veval')
        self.V_targ = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name='Vtarg')

    def save(self):
        self.Q.model.save_weights(self.name)

    def process_board(self,obs,conf,gindex):
        rows, columns = conf.rows, conf.columns
        board = np.zeros((rows+2,columns+2,3))
        if 'food' in obs.keys():
            for food in obs['food']:
                r,c = row_col(food,columns)
                for i in range(r,r+3):
                    for j in range(c,c+3):
                        board[i,j,0] = 1.0 if i == r+1 and j == c+1 else 0.0
        for gind, goose in enumerate(obs['geese']):
            if len(goose) > 0:
            
                for v in goose[1:-1]:
                    r,c = row_col(v,columns)
                    board[r+1,c+1,1] = 0.66
                r,c = row_col(goose[-1],columns)
                board[r+1,c+1,1] = 0.1
            
                if gind != gindex:
                    r,c = row_col(goose[0],columns)
                    board[r+1,c+1,1] = 1.0
                    for i in [0,2]:
                        if board[r+i,c+1,1] == 0: board[r+i,c+1,1] = 0.33
                        if board[r+1,c+i,1] == 0: board[r+1,c+i,1] = 0.33
                else:
                    for v in goose[1:-1]:
                        r,c = row_col(v,columns)
                        board[r+1,c+1,2] = 0.66
                    r,c = row_col(goose[0],columns)
                    board[r+1,c+1,2] = 1.0

        for i in range(rows):
            if board[i+1,-2,1] == 1.0: board[i+1,1,1] = 0.33
            if board[i+1,1,1] == 1.0: board[i+1,-2,1] = 0.33
        for i in range(columns):
            if board[-2,i+1,1] == 1.0: board[1,i+1,1] = 0.33
            if board[1,i+1,1] == 1.0: board[-2,i+1,1] = 0.33

        board[0] = board[-2]
        board[-1] = board[1]
        board[:,0] = board[:,-2]
        board[:,-1] = board[:,1]
        board[0,0] = board[-2,-2]
        board[0,-1] = board[-2,1]
        board[-1,0] = board[1,-2]
        board[-1,-1] = board[1,1]
                
        return board

    def display_batch(self,conf,bs=4):
        state,action,reward,done,state_ = self.M.get_batch(bs)
        for i in range(bs):
            F = plt.figure()
            ax = F.add_subplot(121)
            ax.imshow(state[i])
            [ax.axvline(1.5+_,color='k') for _ in range(conf.columns)]
            [ax.axhline(1.5+_,color='k') for _ in range(conf.rows)]
            [ax.axhline(_,color='w') for _ in [0.5,conf.rows+.5]]
            [ax.axvline(_,color='w') for _ in [0.5,conf.columns+.5]]
            ax.set_title(f'Action: {self.actions[action[i]]}')
            ax = F.add_subplot(122)
            ax.imshow(state_[i])
            [ax.axvline(1.5+_,color='k') for _ in range(conf.columns)]
            [ax.axhline(1.5+_,color='k') for _ in range(conf.rows)]
            [ax.axhline(_,color='w') for _ in [0.5,conf.rows+.5]]
            [ax.axvline(_,color='w') for _ in [0.5,conf.columns+.5]]
            ax.set_title(f'Reward: {reward[i]}')
            plt.show()

    def choose_action(self,state,length,prev_action,first=False):
        def check_danger(state,action,pa,l):
            if action == self.forbidden[pa] and l < 3:
                return 2.0
            
            r,c = np.where(state[1:-1,1:-1,2] == 1)
            r+=1
            c+=1
            
            if action == 0: r-=1
            if action == 1: r+=1
            if action == 2: c-=1
            if action == 3: c+=1
            if r >= state.shape[0]-1:
                r = 0
            if c >= state.shape[1]-1:
                c = 0
                
            return state[r,c,1]
        
        if np.random.uniform() < self.epsilon:
            done = False
            while not done:
                act = np.random.choice([0,1,2,3])
                if first or length > 2:
                    done = True
                else:
                    if act != self.forbidden[prev_action]:
                        done = True

        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor,0)

            actions = self.Q.model(state_tensor,training=False).numpy()[0]
            acts = np.argsort(actions)[::-1]
            ind = -1
            dangers = np.zeros(4)
            for i in range(4):
                dangers[i] = check_danger(state,acts[i],prev_action,length)
                if dangers[i] == 0 and ind == -1:
                    ind = i
                    
            if ind == -1:
                ind = np.random.choice([0,1,2,3])
                semi = np.where((dangers == 0.33)|(dangers == 0.1))[0]
                if len(semi) > 0:
                    ind = semi[0]

            act = acts[ind]
                    
        self.prev_action=act

        return act

    def learn(self, state, action, reward, done, state_):
        s = np.expand_dims(state,axis=0)
        s_ = np.expand_dims(state_,axis=0)
        V,V_ = tf.reduce_max(self.V_targ.model.predict(s)),tf.reduce_max(self.V_targ.model.predict(s_))
        TD_error = np.abs(reward + self.gamma*V_ - V)
        self.M.store_memory(state, action, reward, done,state_,TD_error)
        
        if self.M.memcount < self.N:
            return

        state,action,reward,done,state_ = self.M.get_batch(self.batch_size)

        V_ = tf.reduce_max(self.V_targ.model.predict(state_),axis=1)
        V_ *= (1.-done)
        yt = reward + self.gamma*V_

        with tf.GradientTape() as Qtape:
            Q = self.Q.model(state)
            Q = tf.reduce_sum(tf.multiply(Q,tf.one_hot(action,self.n_actions)),axis=1)
            Qloss = self.Q.loss(yt,Q)
        Qgrad = Qtape.gradient(Qloss,self.Q.model.trainable_variables)
        self.Q.optimizer.apply_gradients(zip(Qgrad,self.Q.model.trainable_variables))

        with tf.GradientTape() as Vtape:
            V = tf.reduce_max(self.V_eval.model(state),axis=1)
            Vloss = self.V_eval.loss(yt,V)
        Vgrad = Vtape.gradient(Vloss,self.V_eval.model.trainable_variables)
        self.V_eval.optimizer.apply_gradients(zip(Vgrad,self.V_eval.model.trainable_variables))

        self.steps += 1
        if self.epsilon > self.emin: self.epsilon -= self.de
        if self.steps%self.retarget == 0:
            weights = self.V_eval.model.get_weights()
            target_weights = self.V_targ.model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i]
            self.V_targ.model.set_weights(target_weights)
