import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque



class OpticalFlow():
    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 1000,
                               qualityLevel = 0.1,
                               minDistance = 5,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.notGray = False

        self.window = deque(maxlen=3)
        self.winSize = 3

    def point_selection(self,image):
        image = np.uint8(image)
        # Take first frame and find corners in it
        if self.notGray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(image, mask = None, **self.feature_params)



        return p0


    def find_new_points(self,prev_state,state,p0):
        prev_state = np.uint8(prev_state)
        state = np.uint8(state)

        if self.notGray:
            prev_state = cv2.cvtColor(prev_state, cv2.COLOR_BGR2GRAY)
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_state, state, p0, None, **self.lk_params)
        #p1: new positions of input pixels (given through p0)
        #st: boolean: was the optical flow calculated for the given points?
        
        if type(p1)!=np.ndarray:
            print("aaaaa")
            print(type(p1),p1)
            exit()
            return p0,p0


        # Select good points
        p1 = p1[st==1]
        p1 = p1.reshape((p1.shape[0],1,p1.shape[1]))
        p0 = p0[st==1]
        p0 = p0.reshape((p0.shape[0],1,p0.shape[1]))

        return p0,p1

    def draw_vectors(self,state,p0,p1):

        # Create some random colors
        color = [255,0,0]
        # Create a mask image for drawing purposes
        mask = np.zeros_like(np.uint8(prev_state))

        # draw the tracks
        for i,(new,old) in enumerate(zip(p1,p0)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color, 2)
            if self.notGray:
                state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
                frame = cv2.circle(state_rgb,(a,b),2,color,-1)
            else:
                frame = cv2.circle(state,(a,b),2,color,-1)

        try:
            img = cv2.add(frame,mask)
            img = cv2.transpose(img)
            
            cv2.imshow('frame',img)
            k = cv2.waitKey(1)

        except:
            pass

        return k


    def avg_of(self,p0,p1):
        of = 0
        n = 0
        p0 = p0.reshape((p0.shape[0],p0.shape[2]))
        p1 = p1.reshape((p1.shape[0],p1.shape[2]))
        for i in range(p0.shape[0]):
            d = np.sqrt((p1[i][0] - p0[i][0])**2 + (p1[i][1] - p0[i][1])**2)
            of += d
            n += 1

        if n>0 and of>0:
            return of / n
        else:
            return -1


    def optical_flow(self,prev_state,state,action):
        p0 = self.point_selection(prev_state)

        if type(p0) != np.ndarray: return self.window[-1] # if no points to calculate, return last valid value
        
        p0,p1 = self.find_new_points(prev_state,state,p0)

        of = self.avg_of(p0,p1)
        if of==-1: return self.window[-1]
        if action != [1,1]: 
            of = of/5.5

        self.window.append(of)


        if len(self.window) < 3: 
            of = self.window[-1]
        else: 
            of = np.sum(self.window) / self.winSize

        if of > 10: of = 10
        shaped_of = - of / 10

        return shaped_of
    
    def reset(self): 
        self.window = deque(maxlen=self.winSize)