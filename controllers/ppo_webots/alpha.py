import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #cut annoying tf messages

import numpy as np 
import random
import importlib.util
import time
from collections import deque
import cv2
import matplotlib.pyplot as plt
from controller import Keyboard
import pickle
import shelve

from WebotsEnv import *
from utils import *
from ddqn import *
#from curiosity import ICM
import optical_flow as of
from statistics import *




L = Logger()
env = Mitsos()

keyboard = Keyboard() # to control training from keyboard input
keyboard.enable(env.timestep)

observation = env.reset(reset_position=True)
input_dims = [observation[0].shape,observation[1].shape]

agent = Agent(input_dims=input_dims,n_actions=3,lr=0.0001,mem_size=50000)





RESTORE_DAMAGE = 30
n_games = 2000
discrete_actions = [[-1,1],[1,1],[1,-1]]
training = True
epsilon_train = agent.epsilon
k = -1
filename = 'saveforreload.out'




scores = deque(maxlen=100)
i = 0




if os.path.exists(filename):
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
    del my_shelf
    print('VARIABLES LOADED')
    os.remove(filename)



while (i<n_games):

    done = False
    score = 0
    observation = env.reset()
    ep_steps = 0
    print('START GAME',i)
    while not done:

        action_idx = agent.choose_action(observation)
        
        k = keyboard.getKey()
        if k>60: action_idx = k-314

        observation_, reward, done, info = env.step(discrete_actions[action_idx])
        agent.remember(observation,action_idx,reward,observation_,done)
        observation = observation_
        if training: agent.learn()
        score += reward
        ep_steps += 1

        if k == 43:
            training = False
            epsilon_train = agent.epsilon
            agent.epsilon = agent.epsilon_min
            print('Training off')
        if k == 45:
            training = True
            agent.epsilon = epsilon_train
            print('Training on')

    scores.append(score)
    print('EPISODE:',i,'STEPS:',ep_steps,'EPSILON',agent.epsilon,'SCORE:',score,'AVG SCORE:',np.mean(scores))
    agent.save_model()

    i += 1


    if i % RESTORE_DAMAGE == 0:
        myshelve = shelve.open(filename,'n')
        for key in dir():
            try:
                myshelve[key] = globals()[key]
            except TypeError:
                pass
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        myshelve.close()
        del myshelve
        env.robot.worldReload()
