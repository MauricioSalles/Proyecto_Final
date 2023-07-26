import cv2
import numpy as np
import NN_resorces.Dataset as dt
from NN_resorces.refine_flow import FlowModule

dataset = dt.FramesDataset('dataset')
len = dataset.__len__()
flow = FlowModule()
totalError = 0

def calcFrame(frame1 , frame2,flow):
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    flow,flow2 = flow.getFlow(frame1, frame2,segmentType='none',flowType='manual',levels=2)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    newFrame = cv2.remap(frame1, flow,None, cv2.INTER_LINEAR)
    return newFrame

print(len)
for i in range(len):
    print(i)
    f1, f2, f3 = dataset.__getitem__(i)
    fNew = calcFrame(f1,f3,flow)
    error = np.square(fNew-f2).mean()
    totalError += error
print(totalError/len)