import os,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *

import skimage 
from skimage import transform,color,exposure
from skimage.viewer import ImageViewer

import numpy as np
from collections import deque
import cv2


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img


class VizDoomEnv(object):
    def __init__(self,state_size=(64,64,4),scenario='defend_the_center.cfg',record_episode=False):
        game = DoomGame()
        path_to_scenario = os.path.join(current_dir,scenario)
        game.load_config(path_to_scenario)
        game.set_sound_enabled(True)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_window_visible(False)
        game.set_available_game_variables([GameVariable.KILLCOUNT,GameVariable.AMMO2,GameVariable.HEALTH])
        game.init()
        self.game = game
        
        self.skiprate = 4

        self.state = None
        self.state_size = state_size
        self.action_size = self.game.get_available_buttons_size()

        self.steps = 0
        self.life = deque(maxlen=30)
        self.kills = deque(maxlen=30)

        self.record_episode = record_episode
        self.game_rec = []
        
    def reset(self):
        self.game.new_episode()
        game_state = self.game.get_state()
        frame = game_state.screen_buffer # initial resolution 480 x 640
        if self.record_episode: self.game_rec.append(frame)
        frame = preprocessImg(frame, size=(self.state_size[0], self.state_size[1])) # 64x64
        state = np.stack(([frame]*4),axis=2) # 64x64x4 (stack the same frame)
        #self.state = np.expand_dims(state,axis=0) #1x64x64x4
        self.state = state
        self.prev_misc = game_state.game_variables
        #print('new episode')
        return self.state

    def step(self,action_idx):
        # perform action
        action = np.zeros([self.action_size])
        action[action_idx] = 1
        action = action.astype(int).tolist()
        self.game.set_action(action)
        self.game.advance_action(self.skiprate)

        # get state and reward
        done = self.game.is_episode_finished()
        if done: 
            self.kills.append(self.prev_misc[0])
            self.life.append(self.steps)
            #print('LIFE:',self.steps,'KILLS:',self.kills[-1],'AVG-KILLS:',np.mean(self.kills))
            self.steps = 0
            self.game.new_episode()
            
            if len(self.game_rec)>0: 
                self.store_game_rec()
                self.game_rec = []
                self.record_episode = False
        game_state = self.game.get_state()
        reward = self.game.get_last_reward()

        new_frame = game_state.screen_buffer
        if self.record_episode: self.game_rec.append(new_frame)
        misc = game_state.game_variables

        (img_rows, img_cols) = (self.state_size[0], self.state_size[1])
        new_frame = preprocessImg(new_frame, size=(img_rows, img_cols))
        new_frame = np.reshape(new_frame, (img_rows, img_cols, 1))
        self.state = np.append(new_frame, self.state[ :, :, :3], axis=2)

        reward = self.shape_reward(reward,misc,self.prev_misc)
        self.prev_misc = misc
        self.steps += 1
        return self.state,reward,done,(self.kills[-1] if done else 0)

    def shape_reward(self, r_t, misc, prev_misc):

        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]): # Loss HEALTH
            r_t = r_t - 0.1

        return r_t


    def store_game_rec(self):
        
        filename = 'vizdoom_rec_0'
        filenames = os.listdir(current_dir) # returns list
        while(True):
            if filename in filenames:
                filename = filename[:-1] + str(int(filename[-1])+1)
            else:
                break
            
        path = os.path.join(current_dir,filename)
        keep_variables = np.array(self.game_rec,dtype=object)
        f = open(path,'wb')
        np.save(f,keep_variables)
        f.close()

        # img_array = self.game_rec
        # image = (img_array[0]*255).astype(np.uint8)
        # size = image.shape[:2]
        # out = cv2.VideoWriter(filename+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize=(size[1],size[0]) )
         
        # for i in range(len(img_array)):
        #     image = img_array[i]
        #     image = (image*255).astype(np.uint8)
        #     out.write(image)
        # out.release()
        
