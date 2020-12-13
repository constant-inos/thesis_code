from tensorflow.keras.layers import Input,Dense,Flatten, Conv2D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import os
import random

def build_dqn(lr,n_actions,input_dims):

    input_ = Input(input_dims[0])
    input2 = Input(input_dims[1])
    conv_1 = Conv2D(64,kernel_size=9,activation='relu')(input_)
    conv_2 = Conv2D(64,kernel_size=5,activation='relu')(conv_1)
    conv_3 = Conv2D(64,kernel_size=3,activation='relu')(conv_2)
    flaten = Flatten()(conv_3)
    concat = Concatenate(axis=-1)([flaten,input2])
    dense1 = Dense(512,activation='relu')(concat)
    dense2 = Dense(256,activation='relu')(dense1)
    output = Dense(n_actions)(dense2)

    dqn = Model(inputs=[input_,input2],outputs=[output])
    dqn.compile(optimizer=Adam(lr=lr), loss='mse')
    return dqn


class Agent(object):
    def __init__(self,input_dims,n_actions,lr=0.0001,
            gamma=0.99,batch_size=32,epsilon_max=1.0,epsilon_min=0.0001,
            mem_size=50000,fname='dqn_model.h5'):

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma 
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname
        self.train_intervals = 100
        self.update_target_freq = 3000

        self.dqn = build_dqn(lr,n_actions,input_dims)
        self.target_dqn = build_dqn(lr,n_actions,input_dims)
        self.load_model()

        self.memory = deque(maxlen=mem_size)
        self.mem_cntr = 0

    def remember(self,state,action_idx,reward,new_state,done):
        self.memory.append((state,action_idx,reward,new_state,1-int(done)))
        self.mem_cntr += 1

    def sample_memory(self,n_samples):
        samples = random.sample(self.memory,n_samples)
        
        states = np.zeros((n_samples,)+self.input_dims[0])
        sensors = np.zeros((n_samples,)+self.input_dims[1])
        action_ind = np.zeros((n_samples,))
        rewards = np.zeros((n_samples,))
        new_states = np.zeros((n_samples,)+self.input_dims[0])
        new_sensors = np.zeros((n_samples,)+self.input_dims[1])
        notdones = np.zeros((n_samples,))

        for i in range(n_samples):
            states[i] = samples[i][0][0]
            sensors[i] = samples[i][0][1]
            action_ind[i] = samples[i][1]
            rewards[i] = samples[i][2]
            new_states[i] = samples[i][3][0]
            new_sensors[i] = samples[i][3][1]
            notdones[i] = samples[i][4]

        return [states,sensors],action_ind,rewards,[new_states,new_sensors],notdones

    def choose_action(self,state):
        [state,sensors] = state
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = state.reshape((1,)+state.shape)
            sensors = sensors.reshape((1,)+sensors.shape)
            actions = self.dqn.predict([state,sensors])
            action = np.argmax(actions)
        return action

    def learn(self):

        if self.mem_cntr % self.update_target_freq == 0:
            self.update_target_network()


        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) / 50000

        if not (self.mem_cntr % self.train_intervals == 0): return


        n_samples = min(self.batch_size*self.train_intervals, self.mem_cntr)


        #print('learning')
        [states,sensors],action_ind,rewards,[new_states,new_sensors],notdones = self.sample_memory(n_samples)

        # Action from one-hot-encoding to int
        # action_values = np.array(self.action_space,dtype=np.int8)
        # action_indices = np.dot(action,action_values)


        q_next = self.target_dqn.predict([new_states,new_sensors])
        q_eval = self.dqn.predict([new_states,new_sensors])

        q_pred = self.dqn.predict([states,sensors])
        max_actions = np.argmax(q_eval,axis=1)

        q_target = q_pred

        sample_index = np.arange(n_samples)
        q_target[sample_index,np.argmax(q_target,axis=1)] = rewards[sample_index] + self.gamma*notdones[sample_index]*q_next[sample_index,np.argmax(q_eval,axis=1)]

        self.dqn.fit([states,sensors],q_target,batch_size=self.batch_size,verbose=0)

        return 

    def update_target_network(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def save_model(self):
        self.dqn.save_weights(self.model_file)

    def load_model(self):
        if not os.path.exists(self.model_file): return

        self.dqn.load_weights(self.model_file)
        print('dqn loaded')

        if self.epsilon <= self.epsilon_min:
            self.update_target_network()
