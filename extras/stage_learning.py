import numpy as np
import random


class Goal_Following():
    def __init__(self):
        self.obstacles = []
        self.START = random_position()
        self.GOAL = random_position()

    def random_position():
        x = random.random()*1.9 - 1
        y = random.random()*1.9 - 1
        z = 0
        return [x,y,z]

    def reward_function(dist_from_target,prev_dist_from_target,speed,collision):
        return int(dist_from_target < prev_dist_from_target)*speed - 20*collision + 0.5

    def get_obstacle(object='Chair',pos=[0,0,0]):
        # needs change

        if object=='SolidBox':
            translation = str(pos[1])+' '+str(pos[2]+0.05)+' '+str(pos[0])
            return "SolidBox {  translation "+translation+"  size 0.1 0.1 0.1}"
            
        translation = str(pos[1])+' '+str(pos[2])+' '+str(pos[0])
        scale = objects[object]*3
        proto = "Solid {  translation "+translation+"  scale "+scale+" children [    "+object+" {    }  ]}"
        return proto

    def get_obj_name(self):
        objects = {'Chair':'0.2 ',
           'Ball':'1.6 ',
           'ComputerMouse':'1.4 ',
           'OilBarrel':'0.17 ',
           'Toilet':'0.13 ',
           'SolidBox':''}
#          'LegoWheel':'2.5 ',
        return np.random.choice(list(objects.keys()))

    def create_world(self,SupervisorNode):

        p = self.obstacles()

        for pos in p:
            object = obstacles.get_obj_name()
            nodeString = obstacles.get_obstacle(object,pos)

            root = SupervisorNode.getRoot()
            node = root.getField('children')
            node.importMFNodeFromString(-1,nodeString)




class Simple_Obstacle_Avoidance(Goal_Following):
    def __init__(self):
        super().__init__()

    def get_obstacles(self):
        Ox =  / 2
        Oy = (self.GOAL[1] - self.START[1]) / 2
        Oz = 0
        self.obstacles.append([Ox,Oy,Oz])


class Random_Obstacle_Avoidance(Goal_Following):
    def __init__(self):
        super().__init__()

    def get_obstacles(self,n=3):
        A = (n+1)/10

        [Rx,Ry,_] = self.START
        [Tx,Ty,_] = self.GOAL

        tanfi = (Ty-Ry)/(Tx-Rx)
        fi = np.arctan(tanfi)

        Ax = A/2 * np.cos(fi)
        Ay = A/2 * np.sin(fi)

        for _ in range(n):
            Ox = self.START[0] + 0.1 + (self.GOAL[0] - self.START[0])*random.random()*0.95 + Ax
            Ox = self.START[1] + 0.1 + (self.GOAL[1] - self.START[1])*random.random()*0.95 + Ay
            Oz = 0

        self.obstacles.append([Ox,Oy,Oz])


class Round(Goal_Following):
        def __init__(self):
        super().__init__()
        self.start_places = [[-0.65,-0.8,0],[-0.15,0.75,0],[0.2,-0.75,0],[0.65,-0.7,0],[0.8,-0.2,0],\
             [0.8,0.2,0],[0.55,0.8,0],[0,0.75,0],[-0.3,0.8,0],[-0.75,0.7,0],\
             [-0.8,0.25,0],[-0.8,-0.2,0],[-0.7,-0.75,0]]


    def get_obstacles(self,n=3):
        a = [[x,0.6,0] for x in np.arange(-0.6,0.7,0.1)] + [[x,-0.6,0] for x in np.arange(-0.6,0.7,0.1)] + \
             [[-0.6,y,0] for y in np.arange(-0.5,0.6,0.1)] + [[0.6,y,0] for y in np.arange(-0.5,0.6,0.1)]

        b = [[x,0.95,0] for x in np.arange(-0.95,1.05,0.1)] + [[x,-0.95,0] for x in np.arange(-0.95,1.05,0.1)] + \
             [[-0.95,y,0] for y in np.arange(-0.85,0.95,0.1)] + [[0.95,y,0] for y in np.arange(-0.85,0.95,0.1)]

        o = [[0.5,-0.7,0],[-0.4,-0.73,0],[0,-0.85,0],[-0.3,-0.73,0],[0.8,-0.8,0],[0.7,-0.5,0],\
            [-0.8,-0.8,0],[-0.7,-0.4,0],[-0.85,0,0],[-0.7,0.5,0],\
            [-0.85,0.85,0],[-0.5,0.7,0],[-0.1,0.85,0],[0.3,0.7,0],\
            [0.75,0.85,0],[0.7,0.45,0],[0.85,0,0],[0.7,-0.4,0]]

        self.obstacles = a + b + o 














