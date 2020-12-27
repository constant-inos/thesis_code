import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import os
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from networks.networks import PolicyGradientNetwork


class Agent(object):
    def __init__(self,input_dims,n_actions,lr=0.003,gamma=0.99,fname='reinforce.h5'):
        print('PG4')
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.lr = lr
        self.alpha = 1e-4
        self.gamma = gamma
        self.epCounter = 0
        self.training_interval = 5

        self.G = [] # episodes rewards
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(lr=lr))


    def choose_action(self,observation):
        state = tf.convert_to_tensor([observation])
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]


    def store_experience(self,state,action,reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory)
        rewards = tf.convert_to_tensor(self.reward_memory)

        returns = self.discount_rewards(rewards)

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g,state) in enumerate(zip(returns,self.state_memory)):
                state = tf.convert_to_tensor([state])
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

            gradient = tape.gradient(loss,self.policy.trainable_variables)
            self.policy.optimizer.apply_gradients(zip(gradient,self.policy.trainable_variables))

        self.reward_memory = []
        self.state_memory = []
        self.action_memory = []
        self.epCounter += 1
        return


    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)


    def discount_rewards(self,rewards):
        G = np.zeros_like(rewards)
        r = 0
        for i in reversed(range(len(rewards))):
            r = self.gamma*r + rewards[i]
            G[i] = r

        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        advantages = (G - mean) / std

        return G



if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = Agent(input_dims=4,n_actions=2,lr=0.001,gamma=0.99)

    score_history = []
    episodes = 2000
    avg = []


    for i in range(episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,_ = env.step(action)
            agent.store_experience(observation,action,reward)
            observation = observation_
            score += reward
        score_history.append(score)
        agent.learn()
        w = score_history[-100:]
        print('GAME:',i,'SCORE:',score,'AVG SCORE:',np.mean(w))
        avg.append(np.mean(w))

    plt.plot(avg)
    plt.show()

