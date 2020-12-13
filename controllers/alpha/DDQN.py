from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from networks import DQNetwork

class Agent(object):

    def __init__(self, state_size, action_size, lr=0.0001, batch_size=32, \
                 gamma=0.99, epsilon_max=1.0, epsilon_min=0.0001,\
                 update_target_freq=3000, train_interval=100, \
                 mem_size=50000, fname='dqn.h5'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = [i for i in range(action_size)]
        self.lr = lr
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_freq = update_target_freq
        self.train_interval = train_interval
        self.model_file = fname

        self.memory = deque(maxlen=mem_size)
        self.mem_cntr = 0

        self.model = DQNetwork(action_size)
        self.model.compile(loss='mse',optimizer=Adam(lr))
        self.target_model = DQNetwork(action_size)
    
    def choose_action(self,state):
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([state])
            action = self.model(state).numpy()[0]
            action_idx = np.argmax(action)
        return action_idx

    def sample_memory(self,n_samples):
        samples = random.sample(self.memory,n_samples)
        
        states = np.zeros((n_samples,)+self.state_size)
        action_ind = np.zeros((n_samples,))
        rewards = np.zeros((n_samples,))
        new_states = np.zeros((n_samples,)+self.state_size)
        dones = np.zeros((n_samples,))

        for i in range(n_samples):
            states[i] = samples[i][0]
            action_ind[i] = samples[i][1]
            rewards[i] = samples[i][2]
            new_states[i] = samples[i][3]
            dones[i] = samples[i][4]

        return states,action_ind,rewards,new_states,dones
    
    def store_experience(self,state,action,reward,new_state,done):
        self.memory.append((state,action,reward,new_state,1-int(done)))
        self.mem_cntr += 1

    def learn(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) / 50000
        if self.mem_cntr % self.update_target_freq == 0:
            self.update_target_model()

        if not (agent.mem_cntr % agent.train_interval == 0):
            return

        n_samples = min(self.batch_size*self.train_interval, self.mem_cntr)
        states,action_ind,rewards,new_states,notdones = self.sample_memory(n_samples)

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


if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = Agent(state_size=(4,),action_size=2)

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

        scores.append(score)
        print('GAME:',i,'SCORE:',score,'AVG SCORE:',np.mean(scores[-100:]))
