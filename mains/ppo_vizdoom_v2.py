import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import gym
import tensorflow as tf

import numpy as np
from agents.PPO_v2 import Agent
from environments.VizDoomEnv import VizDoomEnv
from extras.statistics import Logger

env = VizDoomEnv(scenario='defend_the_center.cfg')
agent = Agent(n_actions=env.action_size,conv=True)
L = Logger(name='log_ppo_viz')

TEST_EPOCHS = 5
PPO_STEPS = 256
TARGET_SCORE = 200

train_epochs = 0
early_stop = False

scores = []
while not early_stop:
    observation = env.reset()
    score = 0
    for _ in range(agent.PPO_STEPS):
        action, log_probs, value = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_experience(np.expand_dims(observation,axis=0),action,reward,np.expand_dims(observation_,axis=0),done,log_probs,value)
        score += reward
        if done:
            observation = env.reset()
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('Episode:',len(scores),'Score:',score,'Avg score:',avg_score)
            L.add_log('score',score)
            L.add_log('avg_score',avg_score)
            score = 0
            if avg_score >= 195:
                early_stop = True
                break
            continue
        observation = observation_

    obs = tf.convert_to_tensor([observation_])
    next_value = agent.critic(obs)
    next_value = next_value.numpy()[0][0]
    states,actions,rewards,states_,dones,log_probs,values = agent.read_memory()
    returns = agent.compute_gae(next_value,values,rewards,dones)
    advantages = returns - values
    agent.ppo_update(states,actions,log_probs,returns,advantages)

if train_epochs%20==0:
    L.save_game()
train_epochs += 1
