
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

def D(A,B):
    (x,y) = A
    (a,b) = B
    return np.sqrt((x-a)**2 + (y-b)**2)

def target_reward(P0,P1,TARGET):

    d0 = D(P0,TARGET)
    d1 = D(P1,TARGET)

    # version 1: distance form target
    c = 0.5
    R1 = 1 / (d1+0.1)

    #version 2: difference of distances
    dD = d1-d0
    R2 = -dD*5

    flag = False
    if R2 < 0.0005 and R2 > -0.0005:
        if np.sign(R1)==-1: R1 = -1
    else:
        R1 = 0

    return R1

class Mitsos():
    # Webots-to-environment-agnostic
    def __init__(self,max_steps=2000):
        self.name = "Mitsos"
        self.max_steps = max_steps

        self.robot = Supervisor()    # create the Robot instance
        self.timestep = int(self.robot.getBasicTimeStep())   # get the time step of the current world.
        # crash sensor
        self.bumper = self.robot.getDeviceByIndex(36) #### <---------
        self.bumper.enable(self.timestep)
        
        # camera sensor
        self.camera = self.robot.getCamera("camera")
        self.camera.enable(self.timestep)
        # ir sensors
        IR_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        self.InfraredSensors = [self.robot.getDistanceSensor(s) for s in IR_names]
        for ir in self.InfraredSensors: ir.enable(self.timestep)
        
        # wheels motors
        motors = ["left wheel motor","right wheel motor"]
        self.wheels = []
        for i in range(len(motors)):
            self.wheels.append(self.robot.getMotor(motors[i]))
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0)

        self.robot.step(self.timestep)

        self.cam_shape = (self.camera.getWidth(),self.camera.getHeight())
        self.sensors_shape = (14,)

        self.x_start = -0.71
        self.y_start = -0.83
        self.path = []
        self.map = DynamicMap(self.x_start,self.y_start,map_unit=0.2)
        self.first_step = True
        self.x_target,self.y_target = 0,0
        self.set_target()
        self.misc = [0,0]

        #self.discrete_actions = [[0,-1],[1,0],[0,1]] # normal mode
        #self.discrete_actions = [[1,-1],[1,0],[1,1]] # WebotsRound
        self.discrete_actions = [[0.5,-1],[1,0],[0.5,-1]] # 3rd try
        self.action_size = len(self.discrete_actions)
        self.stepCounter = 0


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
    
    def wait(self,timesteps):
        for _ in range(timesteps):
            self.robot.step(self.timestep)
        return 

    def get_reward(self,version='sparse'):
        reward = 1
        if version == 'sparse':
            if (self.collision()): return reward-100
            #return (not WasVisited) + 1
            return reward + 1

    def create_world(self):

        z = 0
        y = 0.7
        x = -0.7

        s1 = [[x,0.7,0] for x in np.arange(-0.7,0.8,0.1)] + [[x,-0.7,0] for x in np.arange(-0.7,0.8,0.1)] + \
             [[-0.7,y,0] for y in np.arange(-0.6,0.7,0.1)] + [[0.7,y,0] for y in np.arange(-0.6,0.7,0.1)] + \
             [[0.5,-0.8,0],[-0.4,-0.8,0],[0,-0.95,0],[-0.3,-0.8,0],[0.9,-0.9,0],[0.8,-0.5,0]]

        for pos in s1:
            object = obstacles.get_obj_name()
            nodeString = obstacles.get_obstacle(object,pos)

            root = self.robot.getRoot()
            node = root.getField('children')
            node.importMFNodeFromString(-1,nodeString)



    def place_target(self,x,y,z):
        root = self.robot.getRoot()
        node = root.getField('children')
        if not self.first_step: node.removeMF(-1)
        else: self.first_step = False
        translation = 'translation '+str(y)+' '+str(z)+' '+str(x)
        shape = 'Cylinder { height 0.1 radius 0.02}'
        nodeString = " Solid { "+translation+" children [Shape {appearance PBRAppearance {baseColor "+"1 0 0"+" roughness 1 metalness 0} geometry "+shape+" } ]  boundingObject "+shape+"  physics Physics {  }} "
        node.importMFNodeFromString(-1,"DEF My_Solid_"+'TARGET'+nodeString)

    def set_target(self):
        # in analog coordinates
        x = (random.random() - 0.5)*2
        z = 0
        y = (random.random() - 0.5)*2
        #x,y,z = 0,0,0.9
        #self.place_target(x,y,z)
        self.x_target = x
        self.y_target = y

    def reset(self,reset_position=True):
        self.stepCounter = 0
        xs,ys = self.x_start,self.y_start
        self.path = []
        self.map.path = []
        OF.reset()
        self.set_target()
        if (reset_position):
            self.set_position(xs,ys,0.005)  
            self.set_rotation(3.14)
        state,_,_,_ = self.step(1)
        return state


    def step(self,action_idx):
        action = self.discrete_actions[action_idx]
        stacked_frames = 4
        xt,yt = self.x_target,self.y_target
        x0,y0,z = self.get_robot_position()
        was_visited = self.map.visit(x0,y0)
        self.path.append((x0,y0))

        u1,u2 = action
        self.set_wheels_speed(u1,u2)

        xp,yp = x0,y0
        cam4 = np.zeros(shape=self.cam_shape+(4,))
        sensors4 = np.zeros(shape=self.sensors_shape+(4,))
        for i in range(stacked_frames):
            [cam,sensors] = [self.read_camera(),self.read_ir()]
            self.robot.step(self.timestep)
            xn,yn,z = self.get_robot_position()

            pos = [xp,yp,xn,yn,xt,yt]
            xp,yp=xn,yn
            sensors = np.array(list(sensors) + pos)

            cam4[:,:,i] = cam
            sensors4[:,i] = sensors
        
        sensors4 = sensors4.reshape(-1)
        state = [cam4, sensors4]

        xn,yn,z = self.get_robot_position()

        if action==[0,0]:
            return state,0,0,''

        explore = self.map.spatial_std_reward()
        collision = self.collision()
        #r_optic_flow = OF.optical_flow(cam4[:,:,0],cam4[:,:,3],action)
        #r_reach_target = target_reward((x0,y0),(xn,yn),(xt,yt))

        # REWARD FUNCTION 0
        # external_reward = 0.1*explore + -1*collision + 0.1*int(explore > self.misc[0])
        # self.misc = [explore,collision]
        
        # REWARD FUNCTION 1
        # external_reward = -2*collision + 0.4*int(not was_visited)
        # self.misc = [was_visited,collision]
        # if not was_visited: print('New block')
        
        external_reward = -5*collision + 1
        self.misc = [was_visited,collision]
        
        done = collision or (self.stepCounter >= self.max_steps) 
        self.stepCounter += 1
        info = ''
        return state,external_reward,done,info 


    def render():
        return
