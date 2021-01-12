import numpy as np

objects = {'Chair':'0.2 ',
           'Ball':'1.6 ',
           'ComputerMouse':'1.4 ',
           'OilBarrel':'0.17 ',
           'Toilet':'0.13 ',
           'LegoWheel':'2.5 ',
           'SolidBox':''}

def get_obstacle(object='Chair',pos=[0,0,0]):

    if object=='SolidBox':
        translation = str(pos[1])+' '+str(pos[2]+0.05)+' '+str(pos[0])
        return "SolidBox {  translation "+translation+"  size 0.1 0.1 0.1}"
        
    translation = str(pos[1])+' '+str(pos[2])+' '+str(pos[0])
    scale = objects[object]*3
    proto = "Solid {  translation "+translation+"  scale "+scale+" children [    "+object+" {    }  ]}"
    return proto

def get_obj_name():
    return np.random.choice(list(objects.keys()))