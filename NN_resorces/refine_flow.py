import torch
import torch.cuda as cuda
from sklearn.cluster import KMeans
from torch.nn.functional import pad
from torch import load,unsqueeze,cat, no_grad
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
#from torchvision.models.optical_flow import raft_large
#from torchvision.models.optical_flow import Raft_Large_Weights
from torch.nn import DataParallel
from torch import load
try:
    from GMA.core.network import RAFTGMA
    from GMA.core.utils.utils import InputPadder                
except ImportError:
        from .GMA.core.network import RAFTGMA
        from .GMA.core.utils.utils import InputPadder


class FlowModule():
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.ToTensor()
        #self.model = raft_large(weights=Raft_Large_Weights.C_T_V1).to(self.device)
        self.args = self.loadArgs()
        self.flow = DataParallel(RAFTGMA(self.args))
        self.flow.load_state_dict(load(self.args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
        self.flow.eval()
        
    def loadArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default="GMA/checkpoints/gma-things.pth")
        parser.add_argument('--model_name', help="define model name", default="GMA")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--num_heads', default=1, type=int,
                            help='number of heads in attention and aggregation')
        parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
        parser.add_argument('--position_and_content', default=False, action='store_true',
                            help='use position and content-wise attention')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        return parser.parse_args()
    
    def segment(self,image):
        #segment image by color range
        h,w,c = image.shape
        color_index = np.zeros((26,26,26),dtype=int)
        segment_list = []
        colorAbg = []
        segment = 1
        for ht in range(h):
            for wt in range(w):
                b,g,r = (image[ht][wt]+1)//64
                idx = color_index[b][g][r]
                if idx == 0:
                    color_index[b][g][r] = segment
                    idx = segment
                    segment += 1
                    segment_list.append(np.zeros_like(image,dtype=int))
                segment_list[idx-1][ht][wt][0]=b*64
                segment_list[idx-1][ht][wt][1]=g*64
                segment_list[idx-1][ht][wt][2]=r*64
        return segment_list
    
    def segmentCluster(self, image1, image2,  K = 6):
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
            segment = cv2.bitwise_and(res, mask)
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
            relativeFlow = cv2.calcOpticalFlowFarneback(Frame1gray, Frame2gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if i > 0:
                flow = flow + relativeFlow
            else:
                flow = relativeFlow
        return flow
    
    def calcFlowGMA(self, frame1, frame3):
        img1=unsqueeze(self.transform(frame1),0)
        img3=unsqueeze(self.transform(frame3),0)
        padX=0
        padX2=0
        padY=0
        padY2=0
        if w%8 > 0:
            padX = 8-w%8
            if padX%2 == 0:
                padX2 = int(padX/2)
                padX = int(padX/2)
            else:
                padX2 = int(padX/2)
                padX = int(padX/2+1)
        if h%8 >0:
            padY = 8-h%8
            if padY%2 == 0:
                padY2 = int(padY/2)
                padY = int(padY/2)
            else:
                padY2 = int(padY/2)
                padY = int(padY/2+1)
        img1 = pad(img1, (padY2,padY,padX,padX2), "constant", 0)
        img3 = pad(img3, (padY2,padY,padX,padX2), "constant", 0)
        with torch.no_grad():
            flow_low, output = self.flow(img1.to(self.device), img3.to(self.device), iters=20, test_mode=True)
        b,c,w,h = output.shape
        output = output[:,:,padX:w-padX2,padY:h-padY2]
        return np.transpose(output.cpu().numpy()[0], (1,2,0))
        
    def calcSegmentFlow(self, img1, img2):
        #calculate flow of one segment
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            flow_low, flow_up = self.calcFlowGMA(img1, img2)
        return flow_up
    
    def getFlow(self, img1, img2, K=6,segmentType='manual',flowType='manual',levels=1):
        #generate flow based on the calculation of the flows of the segments
        flow12 = []
        flow21 = []
        segments1 = []
        segments2 = []
        Flow12 = []
        Flow21 = []
        if segmentType.__eq__('manual'):
            segments1 = self.segment(img1)
            segments2 = self.segment(img2)
        if segmentType.__eq__('cluster'):  
            segments1, segments2 = self.segmentCluster(img1,img2, K)
        if segmentType.__eq__('none'):
            segments1 = [img1]
            segments2 = [img2]
        for i in range(len(segments1)):
            img1 = segments1[i]
            img2 = segments2[i]
            if flowType.__eq__('manual'):
                f1 = self.calcFlow(img1, img2,levels=levels)
                f2 = self.calcFlow(img2, img1,levels=levels)
            if flowType.__eq__('GMA'):
                f1 = self.calcFlowGMA(img1, img2)
                f2 = self.calcFlowGMA(img2, img1)
            flow12.append((f1))
            flow21.append((f2))
        
        for i in range(len(flow12)):
            if(i == 0):
                Flow12=flow12[i]
                Flow21=flow21[i]
            Flow12 = Flow12 + flow12[i]
            Flow21 = Flow21 + flow21[i]
        return Flow12,Flow21
    