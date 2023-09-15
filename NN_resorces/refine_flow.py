import torch
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch import load


class FlowModule():
    
    def segment(self,image1,image2):
        #segment image by color range
        h,w,c = image1.shape
        color_index = np.zeros((26,26,26),dtype=int)
        segment_list1 = []
        segment_list2 = []
        segment = 1
        for ht in range(h):
            for wt in range(w):
                b1,g1,r1 = (image1[ht][wt]+1)//64
                b2,g2,r2 = (image2[ht][wt]+1)//64
                idx1 = color_index[b1][g1][r1]
                if idx1 == 0:
                    color_index[b1][g1][r1] = segment
                    idx1 = segment
                    segment += 1
                    segment_list1.append(np.zeros_like(image1,dtype=float))
                    segment_list2.append(np.zeros_like(image1,dtype=float))
                idx2 = color_index[b2][g2][r2]
                if idx2 == 0:
                    color_index[b2][g2][r2] = segment
                    idx2 = segment
                    segment += 1
                    segment_list1.append(np.zeros_like(image1,dtype=float))
                    segment_list2.append(np.zeros_like(image1,dtype=float))
                segment_list1[idx1-1][ht][wt][0]=b1*64
                segment_list1[idx1-1][ht][wt][1]=g1*64
                segment_list1[idx1-1][ht][wt][2]=r1*64
                segment_list2[idx2-1][ht][wt][0]=b2*64
                segment_list2[idx2-1][ht][wt][1]=g2*64
                segment_list2[idx2-1][ht][wt][2]=r2*64
        return segment_list1,segment_list2
    
    def segmentCluster(self, image1, image2,  K = 16):
        #segment image by color clusters
        twoDimage = image1.reshape((-1,3))
        twoDimage2 = image2.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        twoDimage2 = np.float32(twoDimage2)
        kmeans = KMeans(n_init='auto',n_clusters=K).fit(twoDimage)
        seg1 = kmeans.predict(twoDimage)
        seg2 = kmeans.predict(twoDimage2)
        seg1 = self.calcSegments(seg1,kmeans.cluster_centers_, image1, K )
        seg2 = self.calcSegments(seg2,kmeans.cluster_centers_, image2, K )
        return seg1, seg2 
     
    def calcSegments(self, label, center, image, K): 
        #generate image based on the clusters
        segments = []
        center = np.uint8(center)
        res = center[label.flatten()]
        res = res.reshape((image.shape))
        labels = label.reshape((image.shape[:-1]))
        for i in range(K):
            mask = cv2.inRange(labels, i, i)
            mask = np.dstack([mask]*3)
            #segment = cv2.bitwise_and(res, mask)
            segment = 255*mask
            segments.append(segment)
        return segments
    
    def calcFlow(self, frame1, frame2, levels = 1):
        framesList1 = [frame1]
        framesList2 = [frame2]
        for i in range(levels-1):
            w,h,c = framesList1[i].shape
            f1 = cv2.pyrDown(framesList1[i],dstsize=(2 // w, 2 // h))
            f2 = cv2.pyrDown(framesList2[i],dstsize=(2 // w, 2 // h))
            framesList1.append(f1)
            framesList2.append(f2)
        framesList1 = framesList1[::-1]
        framesList2 = framesList2[::-1]
        flow = np.zeros_like(framesList1[0])
        for i in range(levels):
            img1 = framesList1[i]
            img2 = framesList2[i]
            w,h,c = img1.shape
            flow = flow[:w,:h,:]
            flow = cv2.pyrUp(np.float32(flow))
            if i > 0:
                flowWarp = flow
                h, w = flow.shape[:2]
                flowWarp[:,:,0] += np.arange(w)
                flowWarp[:,:,1] += np.arange(h)[:,np.newaxis]
                img1 = cv2.remap(img1, flowWarp,None, cv2.INTER_LINEAR)
            Frame1gray = cv2.cvtColor(np.float32(img1), cv2.COLOR_BGR2GRAY)
            Frame2gray = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
            relativeFlow = cv2.calcOpticalFlowFarneback(Frame1gray, Frame2gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
            if i > 0:
                flow = flow + relativeFlow
            else:
                flow = relativeFlow
        return flow
    
        
    
    def getFlow(self, img1, img2, K=6,segmentType='sector',levels=1):
        #generate flow based on the calculation of the flows of the segments
        flow12 = []
        flow21 = []
        segments1 = []
        segments2 = []
        Flow12 = []
        Flow21 = []
        if segmentType.__eq__('sector'):
            segments1,segments2 = self.segment(img1,img2)
        if segmentType.__eq__('cluster'):  
            segments1, segments2 = self.segmentCluster(img1,img2, K)
        if segmentType.__eq__('none'):
            segments1 = [img1]
            segments2 = [img2]
        for i in range(len(segments1)):
            img1 = segments1[i]
            img2 = segments2[i]
            f1 = self.calcFlow(img1, img2,levels=levels)
            f2 = self.calcFlow(img2, img1,levels=levels)
            flow12.append((f1))
            flow21.append((f2))
        
        for i in range(len(flow12)):
            if(i == 0):
                Flow12=flow12[i]
                Flow21=flow21[i]
            Flow12 = Flow12 + flow12[i]
            Flow21 = Flow21 + flow21[i]
        return Flow12,Flow21
    