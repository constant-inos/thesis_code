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

from environments.WebotsEnvC import *
from agents.DDPG import Agent
from extras.experience_memory import *
from networks.networks import *
import __main__
from datetime import datetime

def WithNoise(input_vector):
    mean = 0
    std = 0.005
    n = len(input_vector)
    noise = np.random.normal(mean,std,n)
    return list(np.array(input_vector) + noise)


dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger()

env = Mitsos()
state = env.reset()

keyboard = Keyboard() # to control training from keyboard input
keyboard.enable(env.timestep)

# # USING BOTH CAMERA AND IR SENSORS
# n_inputs = 2
# dqn = ComplexDQN
# mem = MemoryDouble

# USING ONLY IR SENSORS
n_inputs = 1
dqn = SimpleDQN
mem = Memory

# # USING ONLY CAMERA
# n_inputs = 1
# dqn = ConvDQN
# mem = Memory


agent = Agent(action_size=env.action_size, mem_size=50000, Memory=mem, n_inputs=n_inputs, batch_size=64)

# if n_inputs==2:
#     state = [tf.convert_to_tensor([state[0]]),tf.convert_to_tensor([state[1]])]
# else:
#     state = tf.convert_to_tensor([state])

# agent.model(state)
# agent.target_model(state)
# agent.load_model()

RESTORE_DAMAGE = 30
n_games = 500000
training = True

k = -1
filename = os.path.join(parent_dir,'history','checkpoint')

scores = deque(maxlen=100)
goals = deque(maxlen=100)
i = 0

if os.path.exists(filename):
    [agent.memory.memory,agent.memory.memCounter,env.task,i,scores,L.Variables,L.fname,L.time,L.t] = list(np.load(filename,allow_pickle=True))
    env.total_steps = agent.memory.memCounter

while (True):

    done = False
    score = 0
    observation = env.reset()
    ep_steps = 0
    current_time = datetime.now().strftime("%H:%M:%S")
    print('GAME:',i,' - TASK:',env.task,' - CURRENT TIME:',current_time)
    while not done:

        action = agent.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        
        if n_inputs == 2:
            state = [np.expand_dims(observation[0],axis=0),np.expand_dims(observation[1],axis=0)]
            new_state = [np.expand_dims(observation_[0],axis=0),np.expand_dims(observation_[1],axis=0)]
        else:
            state = np.expand_dims(observation,axis=0)
            new_state = np.expand_dims(observation_,axis=0)
            
        agent.store_experience(state,action,reward,new_state)

        observation = observation_
        if training: agent.learn()
        score += reward
        ep_steps += 1
        L.tick()
    
    goal = (reward == 100)
          
    L.add_log('score',score)
    L.add_log('goals',goal)
    L.save_game()
    
    scores.append(score)
    goals.append(goal)

    print('EPISODE:',i,'STEPS:',ep_steps,'SCORE:',score,'AVG SCORE:',np.mean(scores),'goals/100:',sum(goals),'\n')
    agent.save_model()

    i += 1
    score = 0

    if i % RESTORE_DAMAGE == 0:
        keep_variables = [agent.memory.memory,agent.memory.memCounter,env.task,i,scores,L.Variables,L.fname,L.time,L.t]
        keep_variables = np.array(keep_variables,dtype=object)
        f = open(filename,'wb')
        np.save(f,keep_variables)
        f.close()
        
        env.robot.worldReload()
        

print('End of Training!')
