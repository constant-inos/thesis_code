import numpy as np
import random 
from collections import deque

class Memory:
    def __init__(self,n_actions):
        self.memory = []
        self.memCounter = 0
        self.n_actions = n_actions


    def store_experience0(self,state,action,reward,state_,done,log_prob,value):
        self.memory.append((state,action,reward,state_,done,log_prob,value)) 
        self.memCounter += 1
    
    def store_experience(self,*experience):
        self.memory.append(experience)
        self.memCounter += 1

    def read_memory(self):
        samples = self.memory
        batch_size = self.memCounter
        
        num_lists = len(self.memory[0])
        lists = [[] for _ in range(num_lists)]
        
        for sample in samples:
            for i in range(len(sample)):
                lists[i].append(sample[i])

        
        lists = [np.vstack(l) if isinstance(l[0],np.ndarray) else np.vstack(l).reshape(-1) for l in lists ]

        return tuple(lists)

    def clear(self):
        self.__init__(self.n_actions)



    # def read_memory0(self):
    #     samples = self.memory
    #     batch_size = self.memCounter

    #     states = np.zeros((batch_size,)+self.state_shape)
    #     actions = np.zeros((batch_size,))
    #     rewards = np.zeros((batch_size,))
    #     states_ = np.zeros((batch_size,)+self.state_shape)
    #     dones = np.zeros((batch_size,))
    #     log_probs = np.zeros((batch_size,))
    #     values = np.zeros((batch_size,))

    #     for i in range(batch_size):
    #         states[i,:] = samples[i][0]
    #         actions[i] = samples[i][1]
    #         rewards[i] = samples[i][2]
    #         states_[i,:] = samples[i][3]
    #         dones[i] = samples[i][4]
    #         log_probs[i] = samples[i][5]
    #         values[i] = samples[i][6]

    #     return states,actions,rewards,states_,dones,log_probs,values
