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
from datetime import datetime

class Memory(Memory):
    def __init__(self,n_actions):
        super(Memory,self).__init__(n_actions=n_actions)

    def sample_memory(self,n_samples):
        samples = random.sample(self.memory,n_samples)
        batch_size = self.memCounter
        
        num_lists = len(self.memory[0])
        lists = [[] for _ in range(num_lists+2)]
        
        for sample in samples:
            image = sample[0][0]
            image_ = sample[3][0]
            sensors = sample[0][1]
            sensors_ = sample[3][1]
            sample = [image,sensors] + list(sample[1:3]) + [image_,sensors_] + list(sample[4:])
            for i in range(len(sample)):
                lists[i].append(sample[i])

        
        lists = [np.vstack(l) if isinstance(l[0],np.ndarray) else np.vstack(l).reshape(-1) for l in lists ]

        lists = [[lists[0],lists[1]]] + lists[2:4] + [[lists[4],lists[5]]] + lists[6:]

        return tuple(lists)

class DoubleInputAgent(Agent):
    def __init__(self, action_size, lr=0.0001, conv=False, batch_size=32, \
                 gamma=0.99, epsilon_max=1.0, epsilon_min=0.0001, epsilon_step=(1/50000),\
                 update_target_freq=3000, train_interval=100, \
                 mem_size=50000, fname='dqn.h5'):
        self.action_size = action_size
        self.action_space = [i for i in range(action_size)]
        self.lr = lr
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_step = epsilon_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_freq = update_target_freq
        self.train_interval = train_interval
        fname = 'network_'+ __main__.__file__.split('.')[0] + '.h5'
        self.model_file = os.path.join(parent_dir,'history',fname)

        self.memory = Memory(n_actions=action_size)

        self.model = MitsosDQNet(action_size)
        self.model.compile(loss='mse',optimizer=Adam(lr))
        self.target_model = MitsosDQNet(action_size)

        
        state = env.reset()
        state = [tf.convert_to_tensor([state[0]]),tf.convert_to_tensor([state[1]])]
        self.model(state)
        self.target_model(state)
        
        self.load_model()


    def choose_action(self,state):

        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            state = [tf.convert_to_tensor([state[0]]),tf.convert_to_tensor([state[1]])]
            action = self.model(state).numpy()[0]
            action_idx = np.argmax(action)
        return action_idx


dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger()

env = Mitsos()

keyboard = Keyboard() # to control training from keyboard input
keyboard.enable(env.timestep)

agent = DoubleInputAgent(action_size=3,lr=0.00025,mem_size=9000)



RESTORE_DAMAGE = 30
n_games = 2000
training = True
epsilon_train = agent.epsilon
k = -1
filename = 'checkpoint'



scores = deque(maxlen=100)
i = 0




if os.path.exists(filename):
    [agent.memory.memory,agent.memory.memCounter,agent.epsilon,env.task,i,scores,L.Variables,L.fname,L.time,L.t] = list(np.load(filename,allow_pickle=True))

    # if i>200 and env.task=='Goal_Following':
    #     env.task = 'Simple_Obstacle_Avoidance'
    #     agent.epsilon = 0.5
    # if i>500 and env.task=='Simple_Obstacle_Avoidance':
    #     env.task = 'Random_Obstacle_Avoidance'
    #     agent.epsilon = 0.5



while (i<n_games):

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

        state = [np.expand_dims(observation[0],axis=0),np.expand_dims(observation[1],axis=0)]
        new_state = [np.expand_dims(observation_[0],axis=0),np.expand_dims(observation_[1],axis=0)]

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
    
    if agent.epsilon < 0.1:
        p = env.path
        p.append(env.GOAL)
        p = np.array(p)
        f = open('path','wb')
        np.save(f,p)
        f.close()

    
    
    print('EPISODE:',i,'STEPS:',ep_steps,'EPSILON',agent.epsilon,'SCORE:',score,'AVG SCORE:',np.mean(scores),'\n')
    agent.save_model()

    i += 1

    if i % RESTORE_DAMAGE == 0:
        keep_variables = [agent.memory.memory,agent.memory.memCounter,agent.epsilon,env.task,i,scores,L.Variables,L.fname,L.time,L.t]
        keep_variables = np.array(keep_variables,dtype=object)
        f = open('checkpoint','wb')
        np.save(f,keep_variables)
        f.close()
        
        env.robot.worldReload()
