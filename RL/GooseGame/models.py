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

# - - - - - #
# Utilities #
#- - - - - -#

def weights_to_txt(model,outname):
    pass

# - - - - - - - #
# Brain Classes #
#- - - - - - - -#

class LibaAgy_mk0():

    def __init__(self,lrate,
                     n_action,
                     dimensions,
                     save_dir='models/',
                     name='GooseLee'):
        
        self.save_file = f'{save_dir}{name}.h5'
        self.optimizer = keras.optimizers.Adam(learning_rate=lrate)
        self.loss = keras.losses.MeanSquaredError()

        model = keras.models.Sequential()
        model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=dimensions))
        model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Dense(n_action,activation='softmax'))

        self.model = model

    def save_model(self):
        self.model.save(self.save_file)

class LibaAgy_lite():

    def __init__(self,lrate,
                     n_action,
                     dimensions,
                     save_dir='models/',
                     name='GooseLee'):
        
        self.save_file = f'{save_dir}{name}.h5'
        self.optimizer = keras.optimizers.Adam(learning_rate=lrate)
        self.loss = keras.losses.MeanSquaredError()

        model = keras.models.Sequential()
        model.add(layers.Conv2D(16,(4,4),activation='relu',padding='same',input_shape=dimensions))
        model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dense(n_action,activation='softmax'))

        self.model = model

    def save_model(self):
        self.model.save(self.save_file)

class PersistenceOfMemory():
    def __init__(self,maxsize,inputshape,n_actions):
        self.memsize = maxsize
        self.memcount = -1
        self.state_memory = np.zeros((maxsize,*inputshape),dtype=np.float32)
        self.next_state_memory = np.zeros((maxsize,*inputshape),dtype=np.float32)
        self.reward_memory = np.zeros(maxsize,dtype=np.float32)
        self.action_memory = np.zeros(maxsize,dtype=np.int64)
        self.terminal_memory = np.zeros(maxsize,dtype=np.float32)

    def store_memory(self,state, action, reward, terminal,state_):
        index = self.memcount%self.memsize
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal 
        self.memcount+=1

    def get_batch(self,batch_size):
        mxind = min(self.memcount,self.memsize)
        batch = np.random.choice(mxind,batch_size,replace=False)

        states = np.array(self.state_memory[batch])
        states_ = np.array(self.next_state_memory[batch])
        rewards = np.array(self.reward_memory[batch])
        actions = np.array(self.action_memory[batch])
        terminals = np.array(self.terminal_memory[batch])

        return states, actions, rewards, terminals, states_
    

# - - - - - #
# 1D Models #
#- - - - - -#

class GooseLee1D_mk0():

    def __init__(self,lrate,brain,
                     n_actions=4,
                     dimensions=(7,11,1),
                     gamma=0.99,
                     epsilon=1.0,
                     e_dec=2.5e-5,
                     e_min=0.01,
                     retarget=100,
                     memsize=10000,
                     memmin=128,
                     batch_size=128,
                     save_dir='models/',
                     name='GooseLee'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.emin = e_min
        self.de = e_dec
        self.action_space = [i for i in range(n_actions)]
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
        
        self.M = PersistenceOfMemory(self.memsize,self.dimensions,self.n_actions)
        self.Q = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name=name)
        self.V_eval = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name='Veval')
        self.V_targ = brain(self.lrate,self.n_actions,self.dimensions,save_dir=save_dir,name='Vtarg')

    def save(self):
        self.Q.model.save_weights(self.name)

    def process_board(self,obs,conf,gindex):
        rows, columns = conf.rows, conf.columns
        board = np.zeros((rows,columns,1))
        if 'food' in obs.keys():
            for food in obs['food']:
                r,c = row_col(food,columns)
                board[r,c] = 1.0
        for gind, goose in enumerate(obs['geese']):
            if len(goose) > 0:
                for v in goose[1:-1]:
                    r,c = row_col(v,columns)
                    board[r,c] = 0.4
                r,c = row_col(goose[-1],columns)
                board[r,c] = 0.2
            
                if gind != gindex:
                    r,c = row_col(goose[0],columns)
                    board[r,c] = 0.6
                else:
                    r,c = row_col(goose[0],columns)
                    board[r,c] = 0.8
        """
        board[1] = board[-3]
        board[0] = board[-4]
        board[-2] = board[2]
        board[-1] = board[3]
        board[:,1] = board[:,-3]
        board[:,0] = board[:,-4]
        board[:,-2] = board[:,2]
        board[:,-1] = board[:,3]
        """
        
        return board

    def display_batch(self,conf,bs=4):
        state,action,reward,done,state_ = self.M.get_batch(bs)
        for i in range(bs):
            F = plt.figure()
            ax = F.add_subplot(121)
            ax.imshow(state[i],cmap=mycm,vmin=0,vmax=1)
            [ax.axvline(.5+_,color='k') for _ in range(conf.columns)]
            [ax.axhline(.5+_,color='k') for _ in range(conf.rows)]
            ax.set_title(f'Action: {self.actions[action[i]]}')
            ax = F.add_subplot(122)
            ax.imshow(state_[i],cmap=mycm,vmin=0,vmax=1)
            [ax.axvline(.5+_,color='k') for _ in range(conf.columns)]
            [ax.axhline(.5+_,color='k') for _ in range(conf.rows)]
            ax.set_title(f'Reward: {reward[i]}')
            plt.show()

    def choose_action(self,state,length,first=False):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_space)

        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor,0)

        actions = self.Q.model(state_tensor,training=False)

        return tf.argmax(actions[0]).numpy()

    def learn(self, state, action, reward, done, state_):
        
        self.M.store_memory(state, action, reward, done, state_)
        
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

# - - - - - #
# 3D Models #
#- - - - - -#

class GooseLee2D_mk0():

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
        
        self.M = PersistenceOfMemory(self.memsize,self.dimensions,self.n_actions)
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
                board[r+1,c+1,0] = 1.0
        for gind, goose in enumerate(obs['geese']):
            if len(goose) > 0:
            
                for v in goose[1:-1]:
                    r,c = row_col(v,columns)
                    board[r+1,c+1,1] = 0.5
                r,c = row_col(goose[-1],columns)
                board[r+1,c+1,1] = 0.25
            
                if gind != gindex:
                    r,c = row_col(goose[0],columns)
                    board[r+1,c+1,1] = 1.0
                else:
                    r,c = row_col(goose[0],columns)
                    board[r+1,c+1,2] = 1.0

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

            actions = self.Q.model(state_tensor,training=False)
            act = tf.argmax(actions[0]).numpy()
            if not first and length <= 2 and act == self.forbidden[prev_action]:
                v, act = tf.nn.top_k(actions[0],2)
                act = act[1].numpy()

        self.prev_action = act

        return act

    def learn(self, state, action, reward, done, state_):
        self.M.store_memory(state, action, reward, done, state_)
        
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

# - - #
# Gym #
#- - -#

class GooseGym():

    def __init__(self,model,nchannel=3,ncheck=10,ndisplay=3.14,n_smart=2,rule_bots=[],switch_epoch=500,border=0,load_weights=None,shuffle=False,save_epochs=100):
        self.env = make('hungry_geese')
        self.env.reset(num_agents=4)
        self.n_epochs = 1000
        self.epoch = 1
        self.ncheck = ncheck
        self.ndisplay = ndisplay
        self.nsmart = n_smart
        self.conf = Configuration(self.env.configuration)
        self.actions = np.array(['NORTH', 'SOUTH', 'WEST', 'EAST'])
        self.step_history = np.zeros(ncheck).astype(int)
        self.goose_steps = [0,0,0,0]
        self.goose = model
        self.rule_bots = [rb(self.conf) for rb in rule_bots]
        self.border = border
        self.shuffle = shuffle
        self.save_epochs = save_epochs
        self.nchannel = nchannel
        self.first = True
        
        if load_weights != None:
            print(f'Loading weights: {load_weights}')
            self.goose.epsilon = 0.01
            self.goose.Q.model.load_weights(load_weights)
            self.goose.V_eval.model.load_weights(load_weights)
            self.goose.V_targ.model.load_weights(load_weights)

        self.prev_length = [1,1,1,1]
        self.prev_act = ['ACTIVE','ACTIVE','ACTIVE','ACTIVE']
        self.prev_action = [None,None,None,None]
        self.best = 0

    def step(self):
        if self.env.done:
            nsteps = self.env.state[0]['observation']['step']
            if nsteps > self.best:
                self.best = nsteps
            self.step_history[self.epoch%self.ncheck] = nsteps
            if self.epoch%self.ncheck == 0:
                print(f'episode {self.epoch} : actions {nsteps} : <last {self.ncheck}> {self.step_history.mean()} : best {self.best} : epsilon {self.goose.epsilon} : steps {self.goose.steps}')
            if self.epoch%self.ndisplay == 0:
                self.goose.display_batch(self.conf)
                
            self.epoch+=1
            self.env.reset(num_agents=4)
            self.prev_length = [1,1,1,1]
            self.prev_act = ['ACTIVE','ACTIVE','ACTIVE','ACTIVE']
            self.prev_action = [None,None,None,None]
            self.first = True
            
        states = np.zeros((4,self.conf.rows+self.border*2,self.conf.columns+self.border*2,self.nchannel))
        rewards = np.zeros(4)
        terminals = np.array([True]*4)
        actions = np.zeros(4).astype(int)
        obs = self.env.state[0]['observation']
        ginds = np.array([0,1,2,3]).astype(int)
        if self.shuffle: np.random.shuffle(ginds)
        for i,gindex in enumerate(ginds):
            if self.env.state[gindex]['status'] == 'ACTIVE':
                states[gindex] = self.goose.process_board(obs,self.conf,gindex)
                if i < self.nsmart:
                    if self.prev_action[gindex] == None:
                        pa = np.random.choice([0,1,2,3])
                    else:
                        pa = list(self.actions).index(self.prev_action[gindex])
                    actions[gindex] = self.goose.choose_action(states[gindex],len(obs['geese'][gindex]),pa,first=self.first)
                else:
                    rbi = gindex%len(self.rule_bots)
                    rbot = self.rule_bots[rbi]
                    actions[gindex] = list(self.actions).index(rbot(obs,gindex))
                self.prev_action[gindex] = self.actions[actions[gindex]]

        self.env.step(list(self.actions[actions]))
        obs = self.env.state[0]['observation']
        for gindex in range(4):
            if self.prev_act[gindex] == 'ACTIVE':
                self.goose_steps[gindex] += 1
                state_ = self.goose.process_board(obs,self.conf,gindex)
                terminal = False
                length = len(obs['geese'][gindex])
                reward = 0.1*length
                if length <= 0:
                    if self.goose_steps[gindex]%self.conf['hunger_rate'] != 0:
                        reward = -100
                    self.prev_act[gindex] = 'DONE'
                else:
                    if length > self.prev_length[gindex]:
                        self.goose_steps[gindex] = 0
                        reward += 100
                    self.prev_length[gindex] = length
                    
            
                self.goose.learn(states[gindex],actions[gindex],reward,terminal,state_)
        self.first = False

    def train(self,nepochs=500):
        f=0
        while self.epoch < nepochs:
            self.step()
            if self.epoch%self.save_epochs == 0 and f == 0:
                print(f'Save at epoch {self.epoch}')
                self.goose.save()
                f = 1
            if self.epoch%self.save_epochs != 0 and f == 1:
                f = 0
