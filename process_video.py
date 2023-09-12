import numpy as np
import cv2
from torch import load,unsqueeze,cat, no_grad
from os.path import exists
from NN_resorces.UNET import UNET
from NN_resorces.refine_flow import refine_flow
import torchvision.transforms as transforms
import torch.cuda as cuda   
import softsplat
import torch
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class process_video(QObject):
    def __init__(self, updateBar,setMaximum,video_dir,name):  
        super().__init__()  
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True   
        self.video_dir = video_dir
        self.name = name
        self.updateBar = updateBar  
        self.setMaximum = setMaximum
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.flow = refine_flow()
        self.model = UNET(in_channels=6,channels=[64, 128, 256, 512]).to(self.device)
        if(exists('./weights/gen-conv.pth')):
            self.model.load_state_dict(load('./weights/gen-conv.pth'))
        self.transform = transforms.ToTensor()
        
        
    def process(self):    
        newVideo = []
        vidcap = cv2.VideoCapture(self.video_dir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setMaximum(frames)
        success,image1 = vidcap.read()
        iter= 0
        self.updateBar(iter)
        while success:
            iter=iter+1
            success,image2 = vidcap.read()
            if success:
                newFrame = self.generateFrame(image1,image2)
                newVideo.append(newFrame)
            self.updateBar(iter)
            cv2.imwrite("frame.jpg", image1)
            frame = cv2.imread("frame.jpg", cv2.IMREAD_UNCHANGED)
            newVideo.append(frame)
            image1 = image2
        self.updateBar(frames)
        height, width, c = newVideo[-1].shape
        video = cv2.VideoWriter(self.name+'.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), fps*2, (width,height))
        print(self.name)
        for i in range(len(newVideo)):
            print(newVideo[i].shape)
            video.write(newVideo[i])
        video.release()
        
    def generateFrame(self, frame1, frame2):
        flow1 = self.flow.calcFlow(frame1, frame2)
        flow2 = self.flow.calcFlow(frame2, frame1)
        fl1=flow1/2
        fl2=flow2/2
        h, w = flow1.shape[:2]
        flow1[:,:,0] += np.arange(w)
        flow1[:,:,1] += np.arange(h)[:,np.newaxis]
        flow2[:,:,0] += np.arange(w)
        flow2[:,:,1] += np.arange(h)[:,np.newaxis]
        tenOne = torch.FloatTensor(np.ascontiguousarray(frame1.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
        tenTwo  = torch.FloatTensor(np.ascontiguousarray(frame2.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
        tenFlow1 = torch.FloatTensor(np.ascontiguousarray(fl1.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenFlow2 = torch.FloatTensor(np.ascontiguousarray(fl2.transpose(2, 0, 1)[None, :, :, :])).cuda()
        frame1Warped = softsplat.softsplat(tenIn=tenOne, tenFlow=tenFlow1, tenMetric=None, strMode='avg') 
        frame2Warped = softsplat.softsplat(tenIn=tenTwo, tenFlow=tenFlow2, tenMetric=None, strMode='avg') 
        print(frame1Warped.shape)
        #frame1Warped = cv2.remap(frame1, flow1,None, cv2.INTER_LINEAR)
        #frame2Warped = cv2.remap(frame2, flow2,None, cv2.INTER_LINEAR)
        #frame1Warped = self.transform(frame1Warped)
        #frame2Warped = self.transform(frame2Warped)  
        #input = unsqueeze(cat([frame1Warped.to(self.device), frame2Warped.to(self.device)], dim=0),0)
        input = cat([frame1Warped.to(self.device), frame2Warped.to(self.device)], dim=1)
        print(input.shape)
        with no_grad():
            output = self.model(input)    
        newFrame = output.cpu().numpy()[0].transpose(1,2,0)*255
        cv2.imwrite("newFrame.jpg", newFrame)
        newFrame = cv2.imread("newFrame.jpg")
        return newFrame