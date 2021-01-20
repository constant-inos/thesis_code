import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from networks.networks import *
from extras.experience_memory import *
import __main__

class Agent(object):

    def __init__(self, action_size, lr=0.0001, conv=False, batch_size=32, \
                 gamma=0.99, epsilon_max=1.0, epsilon_min=0.0001, epsilon_step=1/100000,\
                 update_target_freq=3000, train_interval=100, \
                 mem_size=15000, Network=DQNetwork, Memory=Memory,n_inputs=1):
        
        self.action_size = action_size
        self.action_space = [i for i in range(action_size)]
        self.lr = lr
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_step = epsilon_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_freq = update_target_freq
        self.train_interval = train_interval
        fname = 'network_'+ __main__.__file__.split('.')[0] + '.h5'
        self.model_file = os.path.join(parent_dir,'history',fname)
        self.n_inputs = n_inputs

        self.memory = Memory(n_actions=action_size,memSize=mem_size)

        self.model = Network(action_size)
        self.model.compile(loss='mse',optimizer=Adam(lr))
        self.target_model = Network(action_size)
        
    def choose_action(self,state):
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            if self.n_inputs == 2:
                state = [tf.convert_to_tensor([state[0]]),tf.convert_to_tensor([state[1]])]
            else:
                state = tf.convert_to_tensor([state])
            action = self.model(state).numpy()[0]
            action_idx = np.argmax(action)
        return action_idx
    
    def store_experience(self,state,action,reward,new_state,done):
        self.memory.store_experience(state,action,reward,new_state,1-int(done))

    def learn(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) * self.epsilon_step
        if self.memory.memCounter % self.update_target_freq == 0:
            self.update_target_model()

        if not (self.memory.memCounter % self.train_interval == 0):
            return

        n_samples = min(self.batch_size*self.train_interval, self.memory.memCounter)
        states,action_ind,rewards,new_states,notdones = self.memory.sample_memory(n_samples)

        q_pred = self.model.predict(states)
        q_eval = self.model.predict(new_states)
        q_next = self.target_model.predict(new_states)     
        q_target = q_pred

        sample_index = np.arange(n_samples)
        #q_target[sample_index,np.argmax(q_target,axis=1)] = rewards[sample_index] + self.gamma*notdones[sample_index]*q_next[sample_index,np.argmax(q_eval,axis=1)]
        q_target[sample_index,action_ind.astype(int)] = rewards[sample_index] + self.gamma*notdones[sample_index]*q_next[sample_index,np.argmax(q_eval,axis=1)]

        self.model.fit(states,q_target,batch_size=self.batch_size,verbose=0)

        return

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return

    def save_model(self):
        self.model.save_weights(self.model_file)


    def load_model(self):
        if os.path.exists(self.model_file):
            self.model.load_weights(self.model_file)
            self.target_model.load_weights(self.model_file)
            print('model loaded')

                     
if __name__ == '__main__':
    import gym
    from statistics import *

    env = gym.make('CartPole-v0')
    agent = Agent(action_size=2)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    L = Logger(dir=dir_path,fname='cartpole_ddqn')

    n_games = 2000
    scores = []
    avg_score = 0

    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            new_state,reward,done,_ = env.step(action)
            score += reward
            agent.store_experience(state,action,reward,new_state,done)
            state = new_state

            agent.learn()

            L.tick()

        L.add_log('score',score)
        L.save_game()        
        scores.append(score)
        print('GAME:',i,'SCORE:',score,'AVG SCORE:',np.mean(scores[-100:]))



