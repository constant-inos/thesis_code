import numpy as np 


class cyclic():
    def __init__(self,x,min,max):
        self.min = min
        self.max = max
        self.cycle = max - min
        self.x = self.make_cyclic(x)

    def __add__(self,y):
        r = self.x + y
        return self.make_cyclic(r)

    def __sub__(self,y):
        r = self.x - y
        return self.make_cyclic(r)

    def __mul__(self,y):
        r = self.x * y
        return self.make_cyclic(r)
        
    def make_cyclic(self,r):
        r = r % self.cycle
        if r > self.max:
            r = self.min + (r - self.max)
        elif r < self.min:
            r = self.max + (r - self.min)
        return r
