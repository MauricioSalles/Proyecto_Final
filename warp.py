import numpy as np
import argparse
import cv2

class Warp():
              
    def warpImage(self, image, frame1, frame2):
        Frame1gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        Frame2gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(Frame1gray, Frame2gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)/2
        h, w = flow.shape[:2]
        flow = flow*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(image, flow, None, cv2.INTER_AREA )
        return warped