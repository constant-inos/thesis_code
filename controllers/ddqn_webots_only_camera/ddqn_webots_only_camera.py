import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir) 

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
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#from curiosity import ICM
import extras.optical_flow as of
from extras.statistics import *

from environments.WebotsEnv import *
from agents.DDQN import Agent
from extras.experience_memory import Memory
from networks.networks import *
import __main__


dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger()

env = Mitsos()

keyboard = Keyboard() # to control training from keyboard input
keyboard.enable(env.timestep)

#agent = DoubleInputAgent(action_size=3,lr=0.00025,mem_size=9000)
agent = Agent(action_size=3,lr=0.00025,mem_size=9000,conv=True)



RESTORE_DAMAGE = 30
n_games = 2000
training = True
epsilon_train = agent.epsilon
k = -1
filename = 'checkpoint'



scores = deque(maxlen=100)
i = 0




if os.path.exists(filename):
    [agent.memory.memory,agent.memory.memCounter,agent.epsilon,i,scores,L.Variables,L.fname,L.time,L.t] = list(np.load(filename,allow_pickle=True))
    print(len(agent.memory.memory))
    #double checkpoint
    keep_variables = [agent.memory.memory,agent.memory.memCounter,agent.epsilon,i,scores,L.Variables,L.fname,L.time,L.t]
    keep_variables = np.array(keep_variables,dtype=object)
    f = open('checkpoint_','wb')
    np.save(f,keep_variables)
    f.close()


while (i<n_games):

    done = False
    score = 0
    observation = env.reset()
    ep_steps = 0
    print('START GAME',i)
    while not done:

        action_idx = agent.choose_action(observation[0])
        
        k = keyboard.getKey()
        if k>60: action_idx = k-314

        observation_, reward, done, info = env.step(action_idx)

        state = np.expand_dims(observation[0],axis=0)
        new_state = np.expand_dims(observation_[0],axis=0)

        agent.store_experience(state,action_idx,reward,new_state,done)
        observation = observation_
        if training: agent.learn()
        score += reward
        ep_steps += 1
        L.add_log('reward',reward)
        L.tick()
        if k == 43:
            training = False
            epsilon_train = agent.epsilon
            agent.epsilon = agent.epsilon_min
            print('Training off')
        if k == 45:
            training = True
            agent.epsilon = epsilon_train
            print('Training on')
    L.add_log('score',score)
    L.save_game()
    scores.append(score)
    print('EPISODE:',i,'STEPS:',ep_steps,'EPSILON',agent.epsilon,'SCORE:',score,'AVG SCORE:',np.mean(scores))
    agent.save_model()

    i += 1

    if i % RESTORE_DAMAGE == 0:
        keep_variables = [agent.memory.memory,agent.memory.memCounter,agent.epsilon,i,scores,L.Variables,L.fname,L.time,L.t]
        keep_variables = np.array(keep_variables,dtype=object)
        f = open('checkpoint','wb')
        np.save(f,keep_variables)
        f.close()
        
        env.robot.worldReload()
