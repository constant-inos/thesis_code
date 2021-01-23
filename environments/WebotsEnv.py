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
    if len(A) == 3:
        (x,y,z) = A
        (a,b,c) = B
    else:
        (x,y) = A
        (a,b) = B
    return np.sqrt((x-a)**2 + (y-b)**2)



class Mitsos():
    # Webots-to-environment-agnostic
    def __init__(self,max_steps=500):
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

        self.task = 'Goal_Following'
        self.START = self.random_position()
        self.GOAL = self.random_position()
        self.create_world()


        self.path = []
        self.dists = [2]
        self.map = DynamicMap(self.START[0],self.START[1],map_unit=0.2)
        self.first_step = True
        self.misc = [0,0]

        self.discrete_actions = [[0.5,-1],[1,0],[0.5,1]] # 3rd try
        self.action_size = len(self.discrete_actions)
        self.stepCounter = 0

    def random_position(self):
        x = (random.random()*2 - 1) * 0.95
        y = (random.random()*2 - 1) * 0.95
        z = 0
        return [x,y,z]

    def reward_function(self,dist_from_goal,prev_dist_from_goal,speed,collision):
        
        R_life_is_good = 1
        
        R_collision = - 20*collision
        
        R_reach_goal = int(dist_from_goal < prev_dist_from_goal) - int(not dist_from_goal < prev_dist_from_goal)*5
        if dist_from_goal < min(self.dists[:-1]):
            R_reach_goal += 10 
        return R_reach_goal


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


    def reset(self,reset_position=True):
        
        self.START = self.random_position()
        self.GOAL = self.random_position()
        self.create_world()
        
        self.stepCounter = 0
        self.path = []
        self.dists = [2]
        self.map.path = []
        OF.reset()
        if (reset_position):
            self.set_position(self.START[0],self.START[1],0.005)  
            self.set_rotation(3.14)
        state,_,_,_ = self.step(1)
        return state


    def step(self,action_idx):
        action = self.discrete_actions[action_idx]
        stacked_frames = 4
        [xg,yg,_] = self.GOAL
        x,y,z = self.get_robot_position()
        was_visited = self.map.visit(x,y)
        self.path.append((x,y))

        u1,u2 = action
        self.set_wheels_speed(u1,u2)

        camera_stack = np.zeros(shape=self.cam_shape+(4,))
        sensor_data = []
        position_data = [xg,yg]
        
        for i in range(stacked_frames):
            
            self.robot.step(self.timestep)
            
            cam = self.read_camera()
            camera_stack[:,:,i] = cam
            
            sensors = self.read_ir()
            sensor_data += list(sensors)
            
            x,y,z = self.get_robot_position()
            position_data += [x,y]


        #state = [camera_stack, sensor_data + position_data]
        state = sensor_data + position_data


        # REWARD SIGNALS

        collision = self.collision()
        dist_from_goal = D((x,y),(xg,yg))
        self.dists.append(dist_from_goal)
        #r_optic_flow = OF.optical_flow(cam4[:,:,0],cam4[:,:,3],action)

        # forward speed: action[0]
        # previous distance from goal: self.misc[0]
        reward = self.reward_function(dist_from_goal,self.misc[0],action[0],collision)
        self.misc = [dist_from_goal,collision]
        
        done = collision or (self.stepCounter >= self.max_steps) or (dist_from_goal < 0.05)
        self.stepCounter += 1
        info = ''
        return state,reward,done,info 


    def render(self):
        return



    def create_world(self):

        self.set_obstacle_positions()

        p = self.obstacles

        for pos in p:
            nodeString = self.get_object_proto(pos=pos)

            root = self.robot.getRoot()
            node = root.getField('children')
            node.importMFNodeFromString(-1,nodeString)

    def set_obstacle_positions(self):

        self.obstacles = []

        if self.task == 'Simple_Obstacle_Avoidance':
            Ox = (self.GOAL[0] + self.START[0]) / 2
            Oy = (self.GOAL[1] + self.START[1]) / 2
            Oz = 0
            self.obstacles.append([Ox,Oy,Oz])

        if self.task == 'Random_Obstacle_Avoidance':

            n = int(D(self.START,self.GOAL) // 0.1 // 2)
            n = 3

            x_max = max(self.START[0],self.GOAL[0])
            x_min = min(self.START[0],self.GOAL[0])
            y_max = max(self.START[1],self.GOAL[1])
            y_min = min(self.START[1],self.GOAL[1])

            for _ in range(n):
                Ox = random.random()*(x_max - x_min) + x_min
                Oy = random.random()*(y_max - y_min) + y_min
                Oz = 0

                self.obstacles.append([Ox,Oy,Oz])


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
            return "SolidBox {  translation "+translation+"  size 0.1 0.1 0.1}"
            
        translation = str(pos[1])+' '+str(pos[2])+' '+str(pos[0])
        scale = objects[object]*3
        proto = "Solid {  translation "+translation+"  scale "+scale+" children [    "+object+" {    }  ]}"
        return proto
