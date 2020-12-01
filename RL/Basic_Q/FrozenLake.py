import gym
import matplotlib.pyplot as plt
import numpy as np
from agent import *

env = gym.make('FrozenLake-v0')
Fboi = Agent(env)
n_episode = 10000

rewards = []
win_perc_trail20 = []
indices = []
episodes = np.linspace(0,n_episode-1,n_episode)
epsilons = 0.75*np.exp(((-1.)*episodes)/8000.)
for i_episode in range(n_episode):
    done = False
    observation = env.reset()
    Fboi.state=0
    while not done:
        action = Fboi.Choose_Action()
        next_state, reward, done, info = env.step(action)
        Fboi.Learn(action,reward,next_state)

        if done:
            rewards.append(reward)
            if len(rewards) >= 100:
                last_ten = np.array(rewards[-100:])
                wins = np.where(last_ten == 1.)[0]
                win_perc_trail20.append(float(len(wins))/100.)
                indices.append(i_episode)
                if divmod(len(rewards),250)[1] == 0:
                    Fboi.Render_Qtable()

F = plt.figure()
ax = F.add_subplot(111)
ax.plot(indices,win_perc_trail20,'k-')
plt.show()

