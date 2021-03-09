import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt 
import random
import os
#from IPython.display import clear_output 


def plot_path(filename='episode'):
    filename = os.path.join(parent_dir,'history',filename)
    while(True):
        [path,goal,obstacles] = list(np.load(filename,allow_pickle=True))

        try:
            [path,goal,obstacles] = list(np.load(filename,allow_pickle=True))
        except:
            continue
    

        scale = 1000

        x,y = [],[]
        for i in range(len(path)):
            x.append(int(path[i][0]*scale))
            y.append(int(path[i][1]*scale))

        for i in range(len(obstacles)):
            [xo,yo,_] = obstacles[i]
            xo = int(xo*scale)
            yo = int(yo*scale)
            obstacles[i] = [xo,yo,0]

        goal[0] = int(goal[0]*scale)
        goal[1] = int(goal[1]*scale)



        X = x + [goal[0]] + [o[0] for o in obstacles]
        Y = y + [goal[1]] + [o[1] for o in obstacles]
        xmax = max(X)+20
        xmin = min(X)-20
        ymax = max(Y)+20
        ymin = min(Y)-20

        map = np.zeros((xmax-xmin+1,ymax-ymin+1))
        for i in range(len(path)-1):
            map[x[i]-xmin,y[i]-ymin] = 1

        xg = goal[0] -xmin
        yg = goal[1] -ymin

        for i in range(-10,10):
            for j in range(-10,10):
                try:
                    if xg+i<=0 or yg+j<0: continue
                    map[xg+i,yg+j] = 1
                    k+=1
                except:
                    pass

        xs = x[0] - xmin
        ys = y[0] - ymin
        for i in range(-10,10):
            for j in range(-10,10):
                try:
                    if xs+i<=0 or ys+j<0: continue
                    map[xs+i,ys+j] = 0.5
                except:
                    pass

        for (xo,yo,_) in obstacles:
            xo = xo - xmin
            yo = yo - ymin
            for i in range(-50,50):
                for j in range(-50,50):
                    try:
                        if xo+i<=0 or yo+j<0: continue
                        map[xo+i,yo+j] = 0.5
                    except:
                        pass

        plt.imshow(map)
        plt.pause(2)
        plt.close()
        #clear_output()

plot_path()