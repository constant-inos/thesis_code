import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import random
import tensorflow_probability as tfp

from networks.networks import *
from extras.experience_memory import *

def normalize(x):
    mean = np.mean(x)
    if len(x) == 0:
        std = 1
    else:
        std = np.std(x)
    return (x - mean) / std


class Agent(object):
    def __init__(self,n_actions,lr=0.005,gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.LAMBDA_GAE = 0.95
        self.PPO_EPOCHS = 10
        self.MINIBATCH_SIZE = 64
        self.PPO_EPSILON = 0.15

        self.ppo_network = PPONetwork(n_actions)
        self.ppo_network.compile(optimizer=Adam(lr=lr))

        self.memory = Memory(n_actions=n_actions)

    def choose_action(self,state):
        state = tf.convert_to_tensor([state])
        probs, value = self.ppo_network(state)
        action_dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = action_dist.sample()
        try:
            log_prob = action_dist.log_prob(action)
        except:
            print(state)
            print(action,probs,value)
            print(action_dist)
            
            x = self.ppo_network.fc1(state)
            print(x)
            x = self.ppo_network.fc2(x)
            print(x)
            
            exit()

        return int(action.numpy()[0]), log_prob.numpy()[0], value.numpy()[0][0]

    def read_memory(self):
        memory = self.memory.read_memory()
        self.memory.clear()
        return memory

    def store_experience(self,state,action,reward,state_,done,log_prob,value):
        #self.memory.store_experience(np.expand_dims(state,axis=0),action,reward,np.expand_dims(state_,axis=0),done,log_prob,value)
        self.memory.store_experience(state,action,reward,state_,done,log_prob,value)
        return

    def store_experience1(self,*args):
        Args = []
        for a in args:
            if isinstance(a,np.ndarray): a = np.expand_dims(a,axis=0)
            Args.append(a)
        self.memory.store_experience(tuple(Args))
        return

    def compute_gae(self,next_value, values, rewards, dones):
        values = list(values) + [next_value]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            mask = 1 - int(dones[i])
            delta = rewards[i] + self.gamma*values[i+1]*mask - values[i]
            gae = delta + self.gamma * self.LAMBDA_GAE * gae
            returns.insert(0,gae+values[i])
        return np.array(returns)

    def ppo_iter(self,states,actions,log_probs,returns,advantages):
        # generates random mini-batches until we have covered the full batch
        batch_size = len(states)
        for _ in range(batch_size // self.MINIBATCH_SIZE):
            indices = np.random.randint(0,batch_size,self.MINIBATCH_SIZE)
            yield states[indices],actions[indices],log_probs[indices],returns[indices],advantages[indices]

    def ppo_update(self,states,actions,log_probs,returns,advantages):
        e = self.PPO_EPSILON
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(self.PPO_EPOCHS):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                # grabs random mini-batches several times until we have covered all data
                with tf.GradientTape() as tape:
                    probs,value = self.ppo_network(state)
                    action_dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
                    entropy_loss = tf.math.reduce_mean(action_dist.entropy())

                    new_log_probs = action_dist.log_prob(action) 
                    ratio = tf.math.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantage
                    surr2 = tf.clip_by_value(ratio,1-e,1+e)*advantage

                    actor_loss = -tf.math.minimum(surr1,surr2)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    critic_loss = tf.math.pow(return_ - value,2)

                    c1 = 0.1
                    c2 = -0.05
                    loss = actor_loss + c1*critic_loss + c2*entropy_loss
                gradients = tape.gradient(loss,self.ppo_network.trainable_variables)
                self.ppo_network.optimizer.apply_gradients(zip(gradients,self.ppo_network.trainable_variables))



if __name__ == '__main__':
    import gym

        
    env = gym.make('CartPole-v0')
    agent = Agent(n_actions=2)

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
            agent.store_experience(np.expand_dims(observation,axis=0),action,reward,np.expand_dims(observation_,axis=0),done,log_probs,value)
            if done:
                observation = env.reset()
                continue
            observation = observation_

        obs = tf.convert_to_tensor([observation_])
        _,next_value = agent.ppo_network(obs)
        next_value = next_value.numpy()[0][0]
        states,actions,rewards,states_,dones,log_probs,values = agent.read_memory()
        returns = agent.compute_gae(next_value,values,rewards,dones)
        advantages = returns - values
        agent.ppo_update(states,actions,log_probs,returns,advantages)

        if train_epochs % TEST_EPOCHS == 0:
            score = test_agent(env)
            print(score)
            if score >= TARGET_SCORE:
                early_stop = True

        train_epochs += 1


