import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #cut annoying tf messages
from tensorflow.keras.optimizers import Adam
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp

from networks.networks import *
from extras.statistics import Logger


class Agent(object):
    def __init__(self, n_actions=2,lr=0.01, gamma=0.99):
        self.lr=lr
        self.gamma=gamma
        self.n_actions=n_actions
        self.action_space = [i for i in range(n_actions)]

        # self.actor_critic = SimpleACNet(n_actions=n_actions)
        # self.actor_critic.compile(Adam(lr=lr))
        self.actor = Actor(n_actions=n_actions)
        self.actor.compile(Adam(lr=0.00005))
        self.critic = Critic(n_actions=n_actions)
        self.critic.compile(Adam(lr=0.0005))

        self.action = None

    def choose_action(self,state):
        state = tf.convert_to_tensor([state])
        probs = self.actor(state)
        action = np.random.choice(self.action_space,p=probs.numpy()[0])
        return action

    def save_model(self):
        return
        #self.actor_critic.save_weights(self.actor_critic.model_name)

    def load_model(self):
        return
        #self.actor_critic.load_weights(self.actor_critic.model_name)

    def learn(self,state,action,reward,state_,done):

        state = tf.convert_to_tensor([state])
        state_ = tf.convert_to_tensor([state_])
        #action = tf.convert_to_tensor([action])
        reward = tf.convert_to_tensor([reward])

        with tf.GradientTape() as tape:
            value = self.critic(state)
            probs = self.actor(state)
            value_ = self.critic(state_)
            probs_ = self.actor(state_)
            value = tf.squeeze(value)
            value_ = tf.squeeze(value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(tf.convert_to_tensor(action))

            """
            log_prob = -sparse_categorical_crossentropy_with_logits
            what is the loss function exactly ??? calculate it 
            (how tf works, sess, graph, fast?)
            """

            delta = reward + self.gamma * value_ * (1-int(done)) - value 
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradient,self.actor.trainable_variables))
        
        with tf.GradientTape() as tape:
            value = self.critic(state)
            probs = self.actor(state)
            value_ = self.critic(state_)
            probs_ = self.actor(state_)
            value = tf.squeeze(value)
            value_ = tf.squeeze(value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(tf.convert_to_tensor(action))

            """
            log_prob = -sparse_categorical_crossentropy_with_logits
            what is the loss function exactly ??? calculate it 
            (how tf works, sess, graph, fast?)
            """

            delta = reward + self.gamma * value_ * (1-int(done)) - value 
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradient,self.critic.trainable_variables))

if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = Agent(lr= 0.0001,n_actions=env.action_space.n)
    n_games = 2000

    score_history = []
    max_score, max_avg = 0,0
    L = Logger(name='log_AC_A')

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(obs)
            obs_,reward,done,info = env.step(action)
            score += reward
            agent.learn(obs,action,reward,obs_,done)
            obs = obs_
            steps += 1
        score_history.append(score)
        avg_score  = np.mean(score_history[-100:])

        print('GAMES:',i,'SCORE:',score,'AVG SCORE:',avg_score)
        if i % 100 == 0: print(max_score,max_avg)
        if score > max_score: max_score = score
        if avg_score > max_avg: max_avg = avg_score
        
        
        L.add_log('score',avg_score)
        if i%20==0:
            L.save_game()

        if avg_score > 195:
            L.save_game()
            exit()
