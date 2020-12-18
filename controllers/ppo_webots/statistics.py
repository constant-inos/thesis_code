import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt 
import time
from varname import nameof
import csv
import pandas as pd
from datetime import datetime


class VarLog:
    def __init__(self,name):
        self.name = name
        self.log = []
        self.time = []
        

class Logger:
    def __init__(self):
        self.id = time.time()
        self.Variables = {}
        self.time = -1
        self.vnames = []

    def tick(self,t=1):
        self.time += t

    def add_variable(self,vname):
        self.Variables[vname] = VarLog(vname)

    def add_log(self,vname,value):
        if self.time == -1: return
        if not vname in self.Variables: self.add_variable(vname)

        self.Variables[vname].log.append(value)
        self.Variables[vname].time.append(self.time)

    def add_logs(self,vars):
        for i in range(len(vars)):
            add_log(vnames[i],vars[i])

    def plot(self,var):
        var = self.Variables['var']
        plt.plot(var.time,var.log)
        plt.show()        
        return


    def save_game(self):
        vnames = [vname for vname in self.Variables]
        time = self.Variables['episode'].time
        game = pd.DataFrame(columns=vnames,index=time)
        game['time'] = pd.DataFrame(time,columns=['time'])
        for vname in vnames:
            try:
                log = self.Variables[vname].log
                game[vname] = log
            except:
                print(vname)
        
        now = str(datetime.now())
        path = './stats/'+now+'.csv'
        game.to_csv(path,mode='a')

    def plot_game(self,game,vars):
        for vname in vars:
            plt.plot(game[vname])
        plt.show()

    def get_last_game(self,mypath='./stats'):
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        files.sort()
        path = join(mypath, files[-1])
        game = pd.read_csv(path)
        return game


if __name__ == '__main__':

    L = Logger()
    g = L.get_last_game()
    L.plot_game(g,['r_expand'])