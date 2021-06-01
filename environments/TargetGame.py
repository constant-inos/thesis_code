import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from extras import obstacles
import numpy as np
import random
import cv2


def WithNoise(input_vector):
    mean = 0
    std = 0.005
    n = len(input_vector)
    noise = np.random.normal(mean,std,n)
    return list(np.array(input_vector) + noise)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def D(A,B):
    if len(A) == 3:
        (x,y,z) = A
        (a,b,c) = B
    else:
        (x,y) = A
        (a,b) = B
    return np.sqrt((x-a)**2 + (y-b)**2)

def reward_function(position_data,prev_shaping,collision=False):
    X,Y,X1,Y1 = position_data
    
    reward = 0
    sh1 = -100*(X1**2+Y1**2) 
    shaping = sh1
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    
    done = False
    if collision:
        #reward -= 100
        done = True
    
    if np.sqrt(X1**2+Y1**2) < 3:
        reward = 100
        done = True

    return reward,done,shaping


        
class Follower():
    # Webots-to-environment-agnostic
    def __init__(self,max_steps=50:
        self.max_steps = max_steps

        self.discrete_actions = [0,1,2] 
        self.action_size = len(self.discrete_actions)
        self.stepCounter = 0
        self.shaping = None
        
        self.create_world()

    def reset(self,reset_position=True):
        
        self.create_world()

        self.stepCounter = 0

        self.shaping = None
        
        self.path.append(position)
        state,_,_,_ = self.step(1)
        return state


    def step(self,action):
        
        [xg,yg,] = self.GOAL
        [x0,y0] = self.position
        
        position_data = []
        
        if self.direction == 0: #up
            x1 = x0-1
            if action==1:
                y1 = y0-1
                self.direction=2
            if action==2:
                y1 = y0+1
                self.direction=3
        if self.direction == 1: #down
            x1 = x0+1
            if action==1:
                y1 = y0+1
                self.direction=3
            if action==2:
                y1 = y0-1
                self.direction=2
        if self.direction == 2: #left
            y1 = y0-1
            if action==1:
                x1 = x0+1
                self.direction=1
            if action==2:
                x1=x0-1
                self.direction=0
        if self.direction == 3: #right
            y1 = y0+1
            if action==1:
                x1 = x0-1
                self.direction=0
            if action==2:
                x1=x0+1
                self.direction=1
                
        try:
            self.map[x1,y1] = 1
        except:
            x1,y1 = x1,y0

        position_data = [x0-xg,y0-yg,x1-xg,y1-yg]


        # rho0,phi0 = cart2pol(x-xg,y-yg)
        # rho1,phi1 = cart2pol(x1-xg,y1-yg)
        # state = [rho0,phi0,rho1,phi1]
        state = position_data

        # REWARD
        reward,done,self.shaping = reward_function(position_data,self.shaping)
        
        if reward == 100: print('goal')

        if self.stepCounter >= self.max_steps:
            done = True

        self.path.append([x1,y1])
        self.stepCounter += 1
        info = ''
        return state,reward,done,info 
        

    def create_world(self):
        L = 100
        self.map = np.zeros((L,L))
        self.start = [int(random.random()*L),int(random.random()*L)]
        self.target = [int(random.random()*L),int(random.random()*L)]
        
        self.direction = np.random.choice([1,2,3,4]) # up, down, left, right
        self.position = self.start
        
        
        
