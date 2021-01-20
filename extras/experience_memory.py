import numpy as np
import random 
from collections import deque

class Memory:
    def __init__(self,n_actions,memSize=15000):
        self.memory = []
        self.memory = deque(maxlen=memSize)
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
        
    def sample_memory(self,n_samples):
        samples = random.sample(self.memory,n_samples)
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



class MemoryDouble(Memory):
    def __init__(self,n_actions,memSize=15000):
        super().__init__(n_actions,memSize)
        
    def sample_memory(self,n_samples):
        samples = random.sample(self.memory,n_samples)
        batch_size = self.memCounter
        
        num_lists = len(self.memory[0])
        lists = [[] for _ in range(num_lists+2)]
        
        for sample in samples:
            image = sample[0][0]
            image_ = sample[3][0]
            sensors = sample[0][1]
            sensors_ = sample[3][1]
            sample = [image,sensors] + list(sample[1:3]) + [image_,sensors_] + list(sample[4:])
            for i in range(len(sample)):
                lists[i].append(sample[i])

        
        lists = [np.vstack(l) if isinstance(l[0],np.ndarray) else np.vstack(l).reshape(-1) for l in lists ]

        lists = [[lists[0],lists[1]]] + lists[2:4] + [[lists[4],lists[5]]] + lists[6:]

        return tuple(lists)
