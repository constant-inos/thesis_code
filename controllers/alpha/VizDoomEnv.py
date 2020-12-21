from vizdoom import DoomGame, ScreenResolution
from vizdoom import *

import skimage 
from skimage import transform,color,exposure
from skimage.viewer import ImageViewer

import numpy as np
from collections import deque


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img


class VizDoomEnv(object):
    def __init__(self,state_size=(64,64,4),scenario_path='defend_the_center.cfg'):
        game = DoomGame()
        path_to_scenario = scenario_path
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

    def reset(self):
        self.game.new_episode()
        game_state = self.game.get_state()
        frame = game_state.screen_buffer # initial resolution 480 x 640
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
        game_state = self.game.get_state()
        reward = self.game.get_last_reward()

        new_frame = game_state.screen_buffer
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
