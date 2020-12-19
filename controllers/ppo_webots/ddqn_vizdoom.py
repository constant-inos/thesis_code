import numpy as np
import random
from tensorflow.keras.optimizers import Adam
from VizDoomEnv import *
from networks import *
from experience_memory import *
from DDQN import Agent

env = VizDoomEnv(scenario_path='/content/gdrive/MyDrive/thesis_code/scenarios/defend_the_center.cfg')
agent = Agent(action_size=env.action_size)


n_games = 2000
scores = []
avg_score = 0

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        new_observation,reward,done,_ = env.step(action)
        score += reward
        state = np.expand_dims(observation,axis=0)
        new_state = np.expand_dims(new_observation,axis=0)
        agent.store_experience(state,action,reward,new_state,done)
        observation = new_observation

        agent.learn()

    scores.append(score)
    print('GAME:',i,'SCORE:',score,'AVG SCORE:',np.mean(scores[-100:]))