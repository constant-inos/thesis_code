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
        reward -= 100
        done = True
    
    c=5
    if np.sqrt(X1**2+Y1**2) < c/100:
        reward = 100
        done = True

    return reward,done,shaping
        
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
        self.discrete_actions = [0,1,2]  #[0,1,2,3] 
        self.action_size = len(self.discrete_actions)
        self.stepCounter = 0
        self.substeps = 20
        self.n_obstacles = 2
        self.shaping = None
        self.FIXED_ORIENTATION = False
        self.RELATIVE_ROTATION = True

        self.create_world()

    def reset(self,reset_position=True):

        self.create_world()

        self.stepCounter = 0
        self.path = [(self.START[0],self.START[1])]

        self.set_position(self.START[0],self.START[1],0.005)  
        self.set_orientation(np.random.choice([0,np.pi/2,np.pi,-np.pi/2]))

        self.shaping = None
        state,_,_,_ = self.step(1)
        return state


    def step(self,action_idx):

        [xg,yg,_] = self.GOAL
        x,y,z = self.get_robot_position()
        self.path.append((x,y))

        if self.FIXED_ORIENTATION:
            # Take action
            if action_idx == 0:
                a = 0
            if action_idx == 1:
                a = 90
            if action_idx == 2:
                a = 180
            if action_idx == 3:
                a = -90
            self.turn0(a)
            self.set_wheels_speed(1,0)

        elif self.RELATIVE_ROTATION:
            if action_idx == 0:
                a = -45
            if action_idx == 1:
                a = 0
            if action_idx == 2:
                a = 45
            self.turn(a)
            self.set_wheels_speed(1,0)


        camera_stack = np.zeros(shape=self.cam_shape+(4,))
        sensor_data = []
        
        position_data = []
        

        for i in range(self.substeps):
            self.robot.step(self.timestep)
        x1,y1,z1 = self.get_robot_position()

        collision = self.collision()

        position_data = [x-xg,y-yg,x1-xg,y1-yg]
        sensor_data = list(self.read_ir())


        #state = [camera_stack, sensor_data + position_data]
        state = sensor_data + position_data
        # state = position_data
        # rho0,phi0 = cart2pol(x-xg,y-yg)
        # rho1,phi1 = cart2pol(x1-xg,y1-yg)
        # state = [rho0,phi0,rho1,phi1]

        # REWARD
        reward,done,self.shaping = reward_function(position_data,self.shaping,collision)
        
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

            #self.store_path()

        self.stepCounter += 1
        info = ''

        return state,reward,done,info 
        

    def create_world(self):
        mode = 1
        
        if mode == 0:
            self.GOAL =  [0,0,0]   #self.random_position()
            self.START = [0.2,0.2,0]   #self.random_position()
        
        if mode == 1:
            self.GOAL =  [0,0,0]   #self.random_position()

            d = 0.2
            a = random.random()*np.pi*2
            x,y = pol2cart(d,a)
            self.START = [x,y,0]
        
        while(True):
            try:
                obs = self.robot.getFromDef('OBS')
                obs.remove()
            except:
                break
        
        self.set_obstacle_positions()
        p = self.obstacles

        for pos in p:
            nodeString = self.get_object_proto(pos=pos)

            root = self.robot.getRoot()
            node = root.getField('children')
            node.importMFNodeFromString(-1,nodeString)

    def set_obstacle_positions(self):
        
        n = self.n_obstacles
        self.obstacles = []
        
        while len(self.obstacles) < n:

            d = D(self.GOAL,self.START) * random.random()
            a = random.random()*np.pi*2
            x,y = pol2cart(d,a)
            self.obstacles.append([x,y,0])

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
    
    def set_orientation(self,a):
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

    def turn(self,a):
        phi = np.rad2deg(self.get_robot_rotation()[-1])
        phi1 = phi + a

        w=-2
        w = int(w * np.sign(a))
        self.set_wheels_speed(0,w)

        while(np.abs(phi - phi1)%360 > 5):
            self.robot.step(self.timestep)
            phi = np.rad2deg(self.get_robot_rotation()[-1])


    def turn0(self,a):
        phi = np.rad2deg(self.get_robot_rotation()[-1])
        w = 5

        if phi - a > 180: 
            w = -w

        self.set_wheels_speed(0,w)
        while( np.abs(phi - a) >= 3 ):
            self.robot.step(self.timestep)
            phi = np.rad2deg(self.get_robot_rotation()[-1])


    def get_robot_position(self):
        object = self.robot.getFromDef(self.name)
        y,z,x = object.getPosition()
        return [x,y,z]

    def get_robot_rotation(self):
        object = self.robot.getFromDef(self.name)
        rotationField = object.getField("rotation")  
        a=rotationField.getSFRotation()
        return a

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

        translation = str(pos[1])+' '+str(pos[2]+0.025)+' '+str(pos[0])
        return "DEF OBS SolidBox {  translation "+translation+"  size 0.05 0.05 0.05}"

        # # needs change
        # objects = {'Chair':'0.2 ',
        #    'Ball':'1.6 ',
        #    'ComputerMouse':'1.4 ',
        #    'OilBarrel':'0.17 ',
        #    'Toilet':'0.13 ',
        #    'SolidBox':''}

        # if object=='':
        #     object = np.random.choice(list(objects.keys()))

        # if object=='SolidBox':
        #     translation = str(pos[1])+' '+str(pos[2]+0.05)+' '+str(pos[0])
        #     return "DEF OBS SolidBox {  translation "+translation+"  size 0.05 0.05 0.05}"
            
        # translation = str(pos[1])+' '+str(pos[2])+' '+str(pos[0])
        # scale = objects[object]*3
        # proto = "DEF OBS Solid {  translation "+translation+"  scale "+scale+" children [    "+object+" {    }  ]}"
        # return proto


    def store_path(self):
        filename = 'paths'
        keep_variables = [self.path,self.START,self.GOAL]
        keep_variables = np.array(keep_variables,dtype=object)
        f = open(filename,'a')
        np.save(f,keep_variables)
        f.close()
