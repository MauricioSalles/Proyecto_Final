import torch
from torch.nn.functional import pad
from torch import load,unsqueeze
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torch.nn import DataParallel
from torch import load
try:
    from GMA.core.network import RAFTGMA             
except ImportError:
        from .GMA.core.network import RAFTGMA


class refine_flow():
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.ToTensor()
        self.model = raft_large(weights=Raft_Large_Weights.C_T_V1).to(self.device)
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
           
    def calcFlow(self, frame1, frame3, levels = 1):
        framesList1 = [frame1]
        framesList3 = [frame3]
        for i in range(levels-1):
            w,h,c = framesList1[i].shape
            f1 = cv2.pyrDown(framesList1[i],dstsize=(2 // w, 2 // h))
            f3 = cv2.pyrDown(framesList3[i],dstsize=(2 // w, 2 // h))
            framesList1.append(f1)
            framesList3.append(f3)
        framesList1 = framesList1[::-1]
        framesList3 = framesList3[::-1]
        flow = np.zeros_like(framesList1[0])
        for i in range(levels):
            img1 = framesList1[i]
            img3 = framesList3[i]
            w,h,c = img1.shape
            if i > 0:
                flow = cv2.pyrUp(flow)
                if flow.shape[1]-h > 0:
                    flow = flow[:,:h,:]
                if flow.shape[0]-w > 0:
                    flow = flow[:w,:,:]
                #flow2 = flow
                #h1, w2 = flow2.shape[:2]
                #flow2[:,:,0] += np.arange(h)
                #flow2[:,:,1] += np.arange(w)[:,np.newaxis]
                #framesList1[i] = cv2.remap(framesList1[i], flow2,None, cv2.INTER_AREA )
            img1=unsqueeze(self.transform(img1),0)
            img3=unsqueeze(self.transform(img3),0)
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
                output = self.flow(img1.to(self.device), img3.to(self.device), iters=10)[0]
            b,c,w,h = output.shape
            output = output[:,:,padX:w-padX2,padY:h-padY2]
            relativeFlow = np.transpose(output.cpu().numpy()[0], (1,2,0))
            if i > 0:
                flow = flow + relativeFlow
            else:
                flow = relativeFlow
        return flow