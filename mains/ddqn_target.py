import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
print(parent_dir)


import numpy as np
import random
import os
from tensorflow.keras.optimizers import Adam

from environments.TargetGame import *
from networks.networks import *
from extras.experience_memory import *
from agents.DDQN import Agent
from extras.statistics import *
dir_path = os.path.dirname(os.path.realpath(__file__))
L = Logger()

env = VizDoomEnv(scenario='defend_the_center.cfg')
agent = Agent(action_size=env.action_size,Network=SimpleDQN)

n_games = 2000
scores = []
avg_score = 0

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        new_observation,reward,done,info = env.step(action)
        score += reward
        state = np.expand_dims(observation,axis=0)
        new_state = np.expand_dims(new_observation,axis=0)
        agent.store_experience(state,action,reward,new_state,done)
        observation = new_observation

        agent.learn()

    scores.append(score)
    print('GAME:',i,'epsilon',agent.epsilon,'SCORE:',score,'AVG SCORE:',np.mean(scores[-100:]))
    L.add_log('score',score)

