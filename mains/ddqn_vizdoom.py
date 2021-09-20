import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
#parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir) 
print(parent_dir)


import numpy as np
import random
import os
from tensorflow.keras.optimizers import Adam

from networks.networks import *
from environments.VizDoomEnv import *
from extras.experience_memory import *
from agents.DDQN import Agent
from extras.statistics import *
dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger(name='log_ddqn_viz')


env = VizDoomEnv(scenario='defend_the_center.cfg')
agent = Agent(action_size=env.action_size,batch_size=16,update_target_freq=3000, train_interval=100, \
                 mem_size=15000,Network=ConvDQN,epsilon_step=1/100000)
agent.model_file = 'network_ddqn_vizdoom.h5'

state = env.reset()
state = tf.convert_to_tensor([state])
agent.model(state)
agent.target_model(state)
agent.load_model()

n_games = 2000
scores = []
avg_score = 0
training = False

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        new_observation,reward,done,kills = env.step(action)
        score += reward
        state = np.expand_dims(observation,axis=0)
        new_state = np.expand_dims(new_observation,axis=0)
        agent.store_experience(state,action,reward,new_state,done)
        
        frame = observation#[0]
        L.gameplay.append(frame)
        
        observation = new_observation
        if training: agent.learn()
    
    L.arrays2video(L.gameplay)
    exit()
    
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('GAME:',i,'epsilon',agent.epsilon,'SCORE:',score,'AVG SCORE:',avg_score)
    L.add_log('score',score)
    L.add_log('kills',kills)
    L.add_log('avg_score',avg_score)
    
    print(len(agent.memory.memory))
    if i % 10==0: 
        L.save_game()
        agent.save_model()
        
