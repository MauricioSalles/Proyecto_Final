import cv2
import numpy as np
import NN_resorces.Dataset as dt
from NN_resorces.refine_flow import FlowModule

dataset = dt.FramesDatasetBase('dataset')
len = dataset.__len__()
flow = FlowModule()
totalError = 0
totalErrorCluster = 0
totalErrorSector = 0
totalErrorRefine = 0

def calcFrame(frame1 , frame2,flow,segmentType,levels):
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    flow,flow2 = flow.getFlow(frame1, frame2,segmentType=segmentType,levels=levels)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    newFrame = cv2.remap(frame1, flow,None, cv2.INTER_LINEAR)
    return newFrame


for i in range(100):
    print(i)
    f1, f2, f3 = dataset.__getitem__(99)
    fNewNormal = calcFrame(f1,f3,flow,segmentType='none',levels=1)
    fNewCluster = calcFrame(f1,f3,flow,segmentType='cluster',levels=1)
    fNewSector = calcFrame(f1,f3,flow,segmentType='sector',levels=1)
    fNewRefine = calcFrame(f1,f3,flow,segmentType='none',levels=2)
    fNewRefineSector = calcFrame(f1,f3,flow,segmentType='sector',levels=2)
    totalError = np.str_(np.abs(fNewNormal-f2).mean()/len)
    totalErrorCluster = np.str_(np.abs(fNewCluster-f2).mean()/len)
    totalErrorSector = np.str_(np.abs(fNewSector-f2).mean()/len)
    totalErrorRefine = np.str_(np.abs(fNewRefine-f2).mean()/len)
    totalErrorRefineSector = np.str_(np.abs(fNewRefineSector-f2).mean()/len)
print("error original:  "+ totalError)  
print("error de metodo de cluster:  "+totalErrorCluster)
print("error de metodo de sectores:  "+totalErrorSector)
print("error de metodo de refinado:  "+totalErrorRefine)
print("error de metodo de refinado y sectores:  "+totalErrorRefineSector)