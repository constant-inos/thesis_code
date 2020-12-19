import numpy as np 
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt 
import time
from varname import nameof
import csv
import pandas as pd
from datetime import datetime
import pickle


class VarLog:
    def __init__(self,name):
        self.name = name
        self.log = [] 
        self.time = []    

class Logger:
    def __init__(self,fname):
        self.Variables = {}
        self.fname = fname
        self.time = []
        self.t = -1

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
        if os.path.exists(self.fname):
            os.remove(self.fname)
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

if __name__ == '__main__':
    fname = 'Cartpole_ddqn_0.pkl'
    L = Logger(fname=fname)

    L.load_game(L.fname)

    L.plot('score')