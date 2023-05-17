import numpy as np
import cv2
from NN_resorces.refine_flow import refine_flow
class Warp():
    
    def __init__(self):
        self.refine_flow = refine_flow()
              
    def warpImage(self, image, frame1, frame2):
        flow = self.refine_flow.calcFlow(frame1,frame2)
        h, w = flow.shape[:2]
        flow = -flow*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(image, flow, None, cv2.INTER_AREA )
        return warped