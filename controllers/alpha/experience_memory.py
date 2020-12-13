import numpy as np
import random 
from collections import deque

class Memory:
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

