import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import transforms
import time


class DynamicMap(object):
    def __init__(self,x_start,y_start,map_unit):
        self.map_unit = map_unit
        self.path = []
        self.map = np.ones((1,1))
        self.O = self.discretize(x_start,y_start) # centre of coordinate system

        self.t0 = time.time()

    def discretize(self,x,y):
        return [int(x/self.map_unit),int(y/self.map_unit)]


    def reset(self,x_start,y_start,map_unit):
        self.__init__(x_start,y_start,map_unit)

    def visit(self,xa,ya):
        [xd,yd] = self.discretize(xa,ya) 
        (x,y) = (xd - self.O[0],yd - self.O[1])

        wasVisited = False
        if (x>=0 and x<self.map.shape[0]) and (y>=0 and y<self.map.shape[1]):
            if self.map[x,y]: 
                wasVisited = True 
            self.map[x,y] = 1
            self.path.append((xd,yd))
            return wasVisited
        else:
            x_temp = self.O[0]
            y_temp = self.O[1]
            if x>=0:
                L = np.maximum(x+1,self.map.shape[0])
            else:
                L = -x + self.map.shape[0]
                self.O[0] = xd
            if y>=0:
                W = np.maximum(y+1,self.map.shape[1])
            else:
                W = -y + self.map.shape[1]
                self.O[1] = yd

            new_map = np.zeros((L,W))
            x_prev = x_temp-self.O[0]
            y_prev = y_temp-self.O[1]
            new_map[x_prev:x_prev + self.map.shape[0], y_prev:y_prev+self.map.shape[1]] = self.map
            #time.sleep(5)

            self.map = new_map
            return self.visit(xa,ya)

        if time.time() - self.t0 < 0:
            self.plot_map()
            self.t0 = time.time()

        return


    def plot_map(self):
        plot = self.map.T
        plt.imshow(plot)
        plt.pause(1)
        plt.close()
        return
