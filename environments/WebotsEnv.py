import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from extras.dynamic_map import *
from extras.optical_flow import *

from controller import Robot,Supervisor,Node,Field
from controller import Camera,DistanceSensor,LED,Motor
from extras import obstacles
import numpy as np
import random
import cv2

OF = OpticalFlow()

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
    
    c=5
    if np.sqrt(X1**2+Y1**2) < c/100:
        reward = 100
        done = True

    return reward,done,shaping

class HER():
    def __init__(self):
        self.goal = 0,0
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def reset(self):
        self.goal = 0,0
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self,s,a):
        self.states.append(s)
        self.actions.append(a)

    def in_done(self):
        memory = []
        n = len(self.states)
        prev_shaping = None
        prev_state = None
        prev_action = None
        prev_reward = None
        prev_done = None
        xg,yg = self.states[n-1][2],self.states[n-1][3]
        for i in range(n):
            [x,y,x1,y1] = self.states[i]
            position_data = [x-xg,y-yg,x1-xg,y1-yg]
            
            #state = position_data
            rho0,phi0 = cart2pol(x-xg,y-yg)
            rho1,phi1 = cart2pol(x1-xg,y1-yg)
            state = [rho0,phi0,rho1,phi1]
            
            
            
            reward,done,prev_shaping = reward_function(position_data,prev_shaping)

            done = (i==n-1)
            action = self.actions[i]
            
            if done: reward += 100
            
            if prev_state is not None:
                memory.append([prev_state,prev_action,prev_reward,state,prev_done])
                
                # # Add Gaussian Noise to increase data and regularize
                # memory.append([WithNoise(prev_state),prev_action,prev_reward,WithNoise(state),prev_done])
                # memory.append([WithNoise(prev_state),prev_action,prev_reward,WithNoise(state),prev_done])

            prev_state,prev_action,prev_reward,prev_done = state,action,reward,done
        
        return memory
            
        
class Mitsos():
    # Webots-to-environment-agnostic
    def __init__(self,max_steps=200):
        self.name = "Mitsos"
        self.max_steps = max_steps

        self.robot = Supervisor()    # create the Robot instance
        self.timestep = int(self.robot.getBasicTimeStep())   # get the time step of the current world.
        # crash sensor
        self.bumper = self.robot.getDeviceByIndex(36) #### <---------
        self.bumper.enable(self.timestep)
        
        # camera sensor
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        # ir sensors
        IR_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        self.InfraredSensors = [self.robot.getDevice(s) for s in IR_names]
        for ir in self.InfraredSensors: ir.enable(self.timestep)
        
        # wheels motors
        motors = ["left wheel motor","right wheel motor"]
        self.wheels = []
        for i in range(len(motors)):
            self.wheels.append(self.robot.getDevice(motors[i]))
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0)

        self.robot.step(self.timestep)

        self.cam_shape = (self.camera.getWidth(),self.camera.getHeight())
        self.sensors_shape = (14,)

        self.task = "Goal_Following"
        self.discrete_actions = [[0.5,-1],[1,0],[0.5,1]] 
        self.action_size = len(self.discrete_actions)
        self.stepCounter = 0
        self.substeps = 18
        self.shaping = None
        self.misc = [0,0]

        self.create_world()
        self.her = HER()

    def reset(self,reset_position=True):
        
        self.create_world()

        self.stepCounter = 0
        self.path = [(self.START[0],self.START[1])]
        self.her.reset()

        self.set_position(self.START[0],self.START[1],0.005)  
        #theta = self.rotation_to_goal((self.GOAL[0],self.GOAL[1]),(self.START[0]-0.1,self.START[1]),(self.START[0],self.START[1]))
        #self.set_rotation(theta+np.pi)
        self.set_rotation(random.random()*2*3.14)

        self.shaping = None
        state,_,_,_ = self.step(1)
        return state


    def step(self,action_idx):
        action = self.discrete_actions[action_idx]
        [xg,yg,_] = self.GOAL
        x,y,z = self.get_robot_position()
        self.path.append((x,y))

        u1,u2 = action
        self.set_wheels_speed(u1,u2)

        camera_stack = np.zeros(shape=self.cam_shape+(4,))
        sensor_data = []
        
        position_data = []
        
        for i in range(self.substeps):
            self.robot.step(self.timestep)
            if (self.substeps-i)%(self.substeps//4)==0: # we need only 4 samples per step, substeps irrelevant
                # cam = self.read_camera()
                # camera_stack[:,:,i] = cam
                # sensors = self.read_ir()
                # sensor_data += list(sensors)
                x1,y1,z1 = self.get_robot_position()

        x,y,x1,y1,xg,yg = x,y,x1,y1,xg,yg
        position_data = [x-xg,y-yg,x1-xg,y1-yg]


        #state = [camera_stack, sensor_data + position_data]
        #state = sensor_data + position_data
        #state = position_data
        rho0,phi0 = cart2pol(x-xg,y-yg)
        rho1,phi1 = cart2pol(x1-xg,y1-yg)
        state = [rho0,phi0,rho1,phi1]

        # REWARD
        reward,done,self.shaping = reward_function(position_data,self.shaping)
        
        if reward == 100: print('goal')

        if self.stepCounter >= self.max_steps:
            done = True


        if done:
            filename = os.path.join(parent_dir,'history','episode')
            vars = [self.path,self.GOAL,self.obstacles]
            vars = np.array(vars,dtype=object)
            f = open(filename,'wb')
            np.save(f,vars)
            f.close()
        
        
        self.her.add([x,y,x1,y1],action_idx)

        self.stepCounter += 1
        info = ''
        return state,reward,done,info 
        

    def create_world(self):
        mode = 0
        
        if mode == 0:
            self.GOAL =  [0,0,0]   #self.random_position()
            self.START = [0.2,0.2,0]   #self.random_position()
            #if random.random()>0.5: self.START = [-0.2,-0.2,0]
        
        if mode == 1:
            self.GOAL =  [0,0,0]   #self.random_position()
            while(True):
                self.START = self.random_position()
                d = D(self.START,self.GOAL) 
                if d>0.18 and d<0.22:
                    break
    
        # obs = self.robot.getFromDef('OBS')
        # obs.remove()
        
        self.set_obstacle_positions()
        p = self.obstacles

        for pos in p:
            nodeString = self.get_object_proto(pos=pos)

            root = self.robot.getRoot()
            node = root.getField('children')
            node.importMFNodeFromString(-1,nodeString)

    def set_obstacle_positions(self):
        
        n = 0
        self.obstacles = []
        
        while len(self.obstacles) < n:
            [x,y,z] = self.random_position()
            if D(self.START,(x,y,z)) > 0.1 and D(self.GOAL,(x,y,z)) > 0.1:
                self.obstacles.append([x,y,z])

    def render(self):
        return

    def read_ir(self):
        ir_sensors = np.array([ i.getValue() for i in self.InfraredSensors])
        max_ = 2500.0
        for i in range(len(ir_sensors)):
            if ir_sensors[i] < 0: 
                ir_sensors[i] = 0.0
            elif ir_sensors[i] > max_: 
                ir_sensors[i] = 1.0
            else:
                ir_sensors[i] = ir_sensors[i] / max_
        return ir_sensors

    def read_camera(self):
        image = np.uint8(self.camera.getImageArray())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        grayN = gray / 255.0
        return gray
        
    def collision(self):
        return bool(self.bumper.getValue())

    def set_position(self,x,y,z):
        #return
        object = self.robot.getFromDef(self.name)
        positionField = object.getField("translation")  
        Field.setSFVec3f(positionField,[y,z,x])
        for _ in range(5): self.robot.step(self.timestep) # if not nan values in first iteration
    
    def set_rotation(self,a):
        a += np.pi
        object = self.robot.getFromDef(self.name)
        rotationField = object.getField("rotation")  #object.getPosition()
        Field.setSFRotation(rotationField,[0,1,0,a])
        self.robot.step(self.timestep) # if not nan values in first iteration

    def set_wheels_speed(self,u,w):
        # u: velocity
        # w: angular velocity
        u1 = u + w
        u2 = u - w

        self.wheels[0].setVelocity(u1)
        self.wheels[1].setVelocity(u2)

    def get_robot_position(self):
        object = self.robot.getFromDef(self.name)
        y,z,x = object.getPosition()
        return [x,y,z]
        
    def rotation_to_goal(self,G,X1,X2):
        (xg,yg),(x1,y1),(x2,y2) = G,X1,X2
        
        
        if xg == x1:
            theta1 = np.pi/2 + (yg<y1)*np.pi
        else:
            lambda1 = (yg-y1)/(xg-x1)
            if xg > x1:
                theta1 = np.arctan(lambda1)
            else:
                theta1 = np.arctan(lambda1) + np.pi
        

        if x2 == x1:
            theta2 = np.pi/2 + (y2<y1)*np.pi
        else: 
            lambda2 = (y2-y1)/(x2-x1)
            if x2 > x1:
                theta2 = np.arctan(lambda2)
            else:
                theta2 = np.arctan(lambda2) + np.pi

        theta = theta1 - theta2
        
        return theta
        
    
    def wait(self,timesteps):
        for _ in range(timesteps):
            self.robot.step(self.timestep)
        return 

    def random_position(self):
        x = (random.random()*2 - 1) * 0.95 
        y = (random.random()*2 - 1) * 0.95 
        z = 0
        return [x,y,z]

    

    def get_object_proto(self,object='',pos=[0,0,0]):
        # needs change
        objects = {'Chair':'0.2 ',
           'Ball':'1.6 ',
           'ComputerMouse':'1.4 ',
           'OilBarrel':'0.17 ',
           'Toilet':'0.13 ',
           'SolidBox':''}

        if object=='':
            object = np.random.choice(list(objects.keys()))

        if object=='SolidBox':
            translation = str(pos[1])+' '+str(pos[2]+0.05)+' '+str(pos[0])
            return "DEF OBS SolidBox {  translation "+translation+"  size 0.1 0.1 0.1}"
            
        translation = str(pos[1])+' '+str(pos[2])+' '+str(pos[0])
        scale = objects[object]*3
        proto = "DEF OBS Solid {  translation "+translation+"  scale "+scale+" children [    "+object+" {    }  ]}"
        return proto

