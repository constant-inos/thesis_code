import numpy as np
import random
import os
from tensorflow.keras.optimizers import Adam

from statistics import *
dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger(dir=dir_path,fname='vizdoom_ddqn')

from VizDoomEnv import *
from networks import *
from experience_memory import *
from DDQN import Agent

env = VizDoomEnv(scenario_path='defend_the_center.cfg')
agent = Agent(action_size=env.action_size,conv=True)


n_games = 2000
scores = []
avg_score = 0

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
        observation = new_observation

        agent.learn()

    scores.append(score)
    print('GAME:',i,'SCORE:',score,'AVG SCORE:',np.mean(scores[-100:]))
    L.add_log('score',score)
    L.add_log('kills',kills)
    
    if i % 10==0: 
        L.save_game()