import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np 
import os
from os import listdir
#from os.path import isfile, joinimport 
import matplotlib.pyplot as plt 
import time
import pandas as pd
from datetime import datetime
import pickle
import __main__
import pathlib
from datetime import datetime as dt
import extras
import cv2


class VarLog:
    def __init__(self,name):
        self.name = name
        self.log = [] 
        self.time = []    

class Logger:
    def __init__(self,name=''):
        self.Variables = {}
        self.fname = name
        if name=='': 
            self.fname = self.get_fname()
        self.time = []
        self.t = -1
        self.gameplay = []

    def get_fname(self):
        i = 0
        while True:
            fname = 'log_' + __main__.__file__.split('.')[0] +'_'+ str(i)
            fdir = os.path.join(parent_dir,'history',fname)
            if not os.path.exists(fdir):
                return fdir
            p = pathlib.Path(fdir)
            d = dt.timestamp(dt.now()) - p.stat().st_mtime # time since modified
            if d<3600*2:
                return fdir
            i += 1

    def tick(self,t=1):
        self.t += t
        self.time.append(self.t)

    def add_variable(self,vname):
        self.Variables[vname] = VarLog(vname)

    def add_log(self,vname,value):
        if not vname in self.Variables: self.add_variable(vname)
        self.Variables[vname].log.append(value)
        self.Variables[vname].time.append(self.time)

    def add_logs(self,vars):
        if not len(a) == len(vars): print('Error! Wrong number of variables!')

        for i,vname in enumerate(self.Variables):
            add_log(vname,vars[i])

    def save_game(self):
        f = open(self.fname,"wb")
        pickle.dump(self.Variables,f)
        f.close()

    def load_game(self,fname):
        if os.path.exists:
            f1 = open(fname,"rb")
            self.Variables = pickle.load(f1)
            f1.close()
        else:
            print('No such file!')


    def plot_game(self,game,vars):
        for vname in vars:
            plt.plot(game[vname].time,game[vname].log)
        plt.show()

    def plot(self,var):
        var = self.Variables[var]
        plt.plot(var.log)
        plt.show()        
        return


    def arrays2video(self,frames):


        size = frames[0].shape[0],frames[0].shape[1]
        fps = 15

        image = (frames[0]).astype(np.uint8)
        size = image.shape[:-1]
        frame_size = (size[1],size[0]) 

        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MPEG'), fps, frameSize= frame_size)
         
        for i in range(len(frames)):
            image = frames[i][:,:,0]
            # print(image.shape)
            # image = np.rollaxis(image, 0, 3)
            # print(image.shape)
            # exit()
            image = image.astype(np.uint8)
            out.write(image)
        out.release()

if __name__ == '__main__':
    L = Logger(name='test')
    path = os.path.join(parent_dir,'history','log_ddqn_webots_0')
    L.load_game(path)
    L.plot('score')
