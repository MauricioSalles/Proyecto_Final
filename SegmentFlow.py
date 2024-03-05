import cv2
import math
import numpy as np
from sklearn.cluster import KMeans



def segmentCluster( image1, image2,  K = 12):
    #segment image by color clusters
    twoDimage = image1.reshape((-1,3))
    twoDimage2 = image2.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    twoDimage2 = np.float32(twoDimage2)
    kmeans = KMeans(n_clusters=K,n_init='auto').fit(twoDimage)
    seg1 = kmeans.predict(twoDimage)
    seg2 = kmeans.predict(twoDimage2)
    seg1 = calcSegments(seg1,kmeans.cluster_centers_, image1, K )
    seg2 = calcSegments(seg2,kmeans.cluster_centers_, image2, K )
    return np.array(seg1,dtype=np.float32), np.array(seg2,dtype=np.float32) 

def calcSegments( label, center, image, K): 
        #generate image based on the clusters
        segments = []
        center = np.uint8(center)
        res = center[label.flatten()]
        res = res.reshape((image.shape))
        labels = label.reshape((image.shape[:-1]))
        for i in range(K):
            mask = cv2.inRange(labels, i, i)
            segment = (mask*1.0)
            segments.append(segment)
        return segments
    
def calcCenter(mask):
    totalPixels = np.sum(mask)
    h,w = mask.shape
    mask = np.stack([mask,mask],axis=2)
    pos = np.zeros((h,w,2),dtype=np.float32)
    pos[:,:,0] += np.arange(w)
    pos[:,:,1] += np.arange(h)[:,np.newaxis]
    objectPost = pos*mask
    meanX = np.sum(objectPost[:,:,0])/totalPixels
    meanY = np.sum(objectPost[:,:,1])/totalPixels

    return meanX, meanY

def segmentCenters(imageSegmented,segmentsId):
    centers = []   
    for segId in range(1,segmentsId):
        segment = cv2.inRange(imageSegmented, segId, segId)
        meanX, meanY = calcCenter(segment)
        centers.append((int(meanX),int(meanY))) 
    return centers

def objetcSegmentation(image):
    iter = 1000
    id = 1
    image = image*10.0
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (10,10), anchor=(-1, -1), iterations=2)
    for i in range(iter):
        points = np.where(image == 2550.0)
        if len(points[0]) == 0:
            break
        _,image,_,_ = cv2.floodFill(image,None,(points[1][0],points[0][0]),newVal=id)
        id += 1
    return image, id

def CalcMovement(originCenters,destinationCenters):
    displacementX = []
    displacementY = []
    for originId in range(len(originCenters)):
        xOrigin,yOrigin = originCenters[originId]
        finalDestX = xOrigin
        finalDestY = yOrigin
        distance = math.inf
        for destinationId in range(len(destinationCenters)):
            xDest, yDest = destinationCenters[destinationId]
            newDistance = math.sqrt(math.pow((xDest-xOrigin),2)+math.pow((yDest-yOrigin),2))
            if newDistance < distance:
                distance = newDistance
                finalDestX = xDest
                finalDestY = yDest
        displacementX.append(finalDestX-xOrigin)
        displacementY.append((finalDestY-yOrigin))
    return displacementX, displacementY

def calFlowSegment(imageSegmented,segmentsId,centers1,centers2):
    h,w = imageSegmented.shape
    flowX = np.zeros((h,w),dtype=np.float32)
    flowY = np.zeros((h,w),dtype=np.float32)
    displacementX, displacementY = CalcMovement(centers1,centers2)
    for segId in range(1,segmentsId):
        segment = (imageSegmented==segId)*1.0 
        dispX = displacementX[segId-1]
        dispY = displacementY[segId-1]
        flowX += segment * dispX
        flowY += segment * dispY
    return np.stack([flowX,flowY],axis=2)


def calcFow(image1,image2):
    seg1,seg2 = segmentCluster(image1,image2)
    c,h,w = seg1.shape
    flow1 = np.zeros((h,w,2),dtype=np.float32)
    for i in range(c):
        imageSegmented1, segmentsId1 = objetcSegmentation(seg1[i])
        imageSegmented2, segmentsId2 = objetcSegmentation(seg2[i])
        centers1 = segmentCenters(imageSegmented1,segmentsId1)
        centers2 = segmentCenters(imageSegmented2,segmentsId2)
        flow1 += calFlowSegment(imageSegmented1,segmentsId1,centers1,centers2)
    return flow1

