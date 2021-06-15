import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

from networks.networks import *
from extras.experience_memory import *
from extras.statistics import Logger
import __main__




class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)



class Agent(object):
    def __init__(self, action_size=1, actor_lr=0.001, critic_lr=0.002, tau=0.005, batch_size=64,\
             gamma=0.99, upper_bound=2,lower_bound=-2, \
             mem_size=50000, Memory=Memory,n_inputs=1):

        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.memory = Memory(n_actions=action_size,memSize=mem_size)

        self.actor = SimpleDDPG_actor(action_size)
        self.critic = SimpleDDPG_critic(action_size)

        self.target_actor = SimpleDDPG_actor(action_size)
        self.target_critic = SimpleDDPG_critic(action_size)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

    def learn(self):
        
        state, action, reward, next_state = self.memory.sample_memory(min(self.batch_size,self.memory.memCounter))

        if self.memory.memCounter < self.batch_size: return

        state = tf.convert_to_tensor(state)
        action = tf.convert_to_tensor(action.reshape((self.batch_size,self.action_size)))
        reward = tf.convert_to_tensor(reward.reshape((self.batch_size,1)))
        reward = tf.cast(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state)


        self.update(state, action, reward, next_state)


    @tf.function
    def update(self,state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                next_state_batch, target_actions, training=True
            )
            critic_value = self.critic(state_batch, action_batch, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic(state_batch, actions, training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )


    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))


    def choose_action(self,state):
        state = tf.convert_to_tensor([state])    
        action = self.actor(state)

        if self.action_size>1:
            a = np.squeeze(action.numpy())
            an = []
            for a1 in a:
                n1 = self.ou_noise()
                a1n = np.clip(a1+n1,self.lower_bound,self.upper_bound)
                an.append(a1n)

            return an

        else:
            action = action.numpy() + noise         
            action = np.clip(action,self.lower_bound,self.upper_bound)

            return [np.squeeze(action)]



        # action = action.numpy() + noise         
        # action = np.clip(action,self.lower_bound,self.upper_bound)

        # return [np.squeeze(action)]

    def store_experience(self,state,action,reward,new_state):
        self.memory.store_experience(state,action,reward,new_state)


    def save_model(self):
        self.actor.save_weights('actor.h5')
        self.critic.save_weights('critic.h5')
        self.target_actor.save_weights('target_actor.h5')
        self.target_critic.save_weights('target_critic.h5')


    def load_model(self):
        if os.path.exists('actor.h5'):
            self.actor.load_weights('actor.h5')
            self.critic.load_weights('critic.h5')
            self.target_actor.load_weights('target_actor.h5')
            self.target_critic.load_weights('target_critic.h5')
            print('models loaded')


##################### MAIN #################################3

if __name__ == '__main__':
    import gym
    env = gym.make("Pendulum-v0")

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    total_episodes = 100

    agent = Agent(action_size=num_actions,upper_bound=upper_bound,lower_bound=lower_bound)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []


    for ep in range(total_episodes):

        state = env.reset()
        episodic_reward = 0

        while True:

            action = agent.choose_action(state)
            
            new_state, reward, done, info = env.step(action)
            
            agent.memory.store_experience(state, action, reward, new_state)
            episodic_reward += reward

            agent.learn()
            agent.update_target(agent.target_actor.variables, agent.actor.variables)
            agent.update_target(agent.target_critic.variables, agent.critic.variables)

            # End this episode when `done` is True
            if done:
                break

            state = new_state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()



        