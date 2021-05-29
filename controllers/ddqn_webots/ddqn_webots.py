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
from extras.experience_memory import *
from networks.networks import *
import __main__
from datetime import datetime


dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger()

env = Mitsos()
state = env.reset()

keyboard = Keyboard() # to control training from keyboard input
keyboard.enable(env.timestep)

n_inputs = 1
agent = Agent(action_size=env.action_size, lr=0.001, mem_size=50000, epsilon_step=1/100000 ,Network=SimpleDQN, Memory=Memory, n_inputs=n_inputs, update_target_freq=30, train_interval=10, batch_size=32)

if n_inputs==2:
    state = [tf.convert_to_tensor([state[0]]),tf.convert_to_tensor([state[1]])]
else:
    state = tf.convert_to_tensor([state])
agent.model(state)
agent.target_model(state)

agent.load_model()

RESTORE_DAMAGE = 30
n_games = 500000
training = True
epsilon_train = agent.epsilon
k = -1
filename = os.path.join(parent_dir,'history','checkpoint')

scores = deque(maxlen=100)
i = 0

if os.path.exists(filename):
    [agent.memory.memory,agent.memory.memCounter,agent.epsilon,env.task,i,scores,L.Variables,L.fname,L.time,L.t] = list(np.load(filename,allow_pickle=True))
    env.total_steps = agent.memory.memCounter

while (True):

    done = False
    score = 0
    observation = env.reset()
    ep_steps = 0
    current_time = datetime.now().strftime("%H:%M:%S")
    print('GAME:',i,' - TASK:',env.task,' - CURRENT TIME:',current_time)
    while not done:

        action_idx = agent.choose_action(observation)
        
        k = keyboard.getKey()
        if k>60: action_idx = k-314

        observation_, reward, done, info = env.step(action_idx)
        
        if n_inputs == 2:
            state = [np.expand_dims(observation[0],axis=0),np.expand_dims(observation[1],axis=0)]
            new_state = [np.expand_dims(observation_[0],axis=0),np.expand_dims(observation_[1],axis=0)]
        else:
            state = np.expand_dims(observation,axis=0)
            new_state = np.expand_dims(observation_,axis=0)

        agent.store_experience(state,action_idx,reward,new_state,done)
        observation = observation_
        if training: agent.learn()
        score += reward
        ep_steps += 1
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
            
    her_memory = env.her.in_done()
    for m in her_memory:
        state,action_idx,reward,new_state,done = m
        state = np.expand_dims(state,axis=0)
        new_state = np.expand_dims(new_state,axis=0)
        agent.store_experience(state,action_idx,reward,new_state,done)
        
    
    L.add_log('score',score)
    L.save_game()
    scores.append(score)


    print('EPISODE:',i,'STEPS:',ep_steps,'EPSILON',agent.epsilon,'SCORE:',score,'AVG SCORE:',np.mean(scores),'\n')
    agent.save_model()

    i += 1

    if i % RESTORE_DAMAGE == 0:
        keep_variables = [agent.memory.memory,agent.memory.memCounter,agent.epsilon,env.task,i,scores,L.Variables,L.fname,L.time,L.t]
        keep_variables = np.array(keep_variables,dtype=object)
        f = open(filename,'wb')
        np.save(f,keep_variables)
        f.close()
        
        env.robot.worldReload()
        
    if agent.epsilon <= agent.epsilon_min:
        break

print('End of Training!')
