import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import gym
import tensorflow as tf

from agents.PPO import Agent
from environments.VizDoomEnv import VizDoomEnv

env = VizDoomEnv(scenario='defend_the_center.cfg')
agent = Agent(state_shape=env.state_size,n_actions=env.action_size)

def test_agent(env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action,_,_ = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        state = state_
        total_reward += reward
    return total_reward

TEST_EPOCHS = 5
PPO_STEPS = 256
TARGET_SCORE = 200

train_epochs = 0
early_stop = False
while not early_stop:
    observation = env.reset()
    for _ in range(PPO_STEPS):
        action, log_probs, value = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_experience(observation,action,reward,observation_,done,log_probs,value)
        if done:
            observation = env.reset()
            continue
        observation = observation_

    obs = tf.convert_to_tensor([observation_])
    _,next_value = agent.ppo_network(obs)
    next_value = next_value.numpy()[0][0]
    states,actions,rewards,states_,dones,log_probs,values = agent.read_memory()#?
    returns = agent.compute_gae(next_value,values,rewards,dones)
    advantages = returns - values
    agent.ppo_update(states,actions,log_probs,returns,advantages)

    if train_epochs % TEST_EPOCHS == 0:
        score = test_agent(env)
        print(score)
        if score >= TARGET_SCORE:
            early_stop = True

    train_epochs += 1
