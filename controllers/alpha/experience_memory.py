import numpy as np
import random 
from collections import deque

class Memory1:
    def __init__(self,state_shape,n_actions,maxlen=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.memCounter = 0
        self.memory = []

        if not maxlen==None:
            self.memory = deque(maxlen=maxlen)
        


    def store_experience(self,states,actions,rewards,states_,dones,log_probs,values):
        self.memory.append((states,actions,rewards,states_,dones,log_probs,values))
        #self.memory.append(experience) 
        self.memCounter += 1

    def read_memory(self):
        if self.memCounter == 0: return None
        samples = self.memory
        batch_size = self.memCounter

        n_lists = len(self.memory[0]) # number of sublists (states,actions,rewards,states_,...)
        lists = [[] for _ in range(n_lists)] # initalize empty sublists

        # List((state,action,reward,...)) --> states,actions,rewards,...
        for i in range(batch_size):
            for l in range(n_lists):
                lists[l].append(samples[i][l]) 

        lists = [np.concatenate([L]) for L in lists]
        #lists = [np.vstack(L) for L in lists]
        return tuple(lists)

    def clear(self):
        self.__init__(self.state_shape,self.n_actions)


 ################################## working
 
class Memory:
    def __init__(self,state_shape,n_actions):
        self.memory = []
        self.memCounter = 0
        self.state_shape = state_shape
        self.n_actions = n_actions


    def store_experience(self,state,action,reward,state_,done,log_prob,value):
        self.memory.append((state,action,reward,state_,done,log_prob,value)) 
        self.memCounter += 1

    def read_memory(self):
        samples = self.memory
        batch_size = self.memCounter

        states = np.zeros((batch_size,)+self.state_shape)
        actions = np.zeros((batch_size,))
        rewards = np.zeros((batch_size,))
        states_ = np.zeros((batch_size,)+self.state_shape)
        dones = np.zeros((batch_size,))
        log_probs = np.zeros((batch_size,))
        values = np.zeros((batch_size,))

        for i in range(batch_size):
            states[i,:] = samples[i][0]
            actions[i] = samples[i][1]
            rewards[i] = samples[i][2]
            states_[i,:] = samples[i][3]
            dones[i] = samples[i][4]
            log_probs[i] = samples[i][5]
            values[i] = samples[i][6]

        return states,actions,rewards,states_,dones,log_probs,values


    def sample_memory(self,batch_size):
        samples = random.sample(self.memory,batch_size)

        states = np.zeros((batch_size,)+self.state_shape)
        actions = np.zeros((batch_size,))
        rewards = np.zeros((batch_size,))
        states_ = np.zeros((batch_size,)+self.state_shape)
        dones = np.zeros((batch_size,))
        log_probs = np.zeros((batch_size,))
        values = np.zeros((batch_size,))

        for i in range(batch_size):
            states[i,:] = samples[i,0]
            actions[i] = samples[i,1]
            rewards[i] = samples[i,2]
            states_[i,:] = samples[i,3]
            dones[i] = samples[i,4]
            log_probs[i] = samples[i,5]
            values[i] = samples[i,6]

        return states,actions,rewards,states_,dones,log_probs,values

    def clear(self):
        self.__init__(self.state_shape,self.n_actions)