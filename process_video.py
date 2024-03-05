import cv2
import argparse
import torchvision.transforms as transforms
import torch.cuda as cuda   
import torch
from torch.nn import MaxPool2d
from torch import load,unsqueeze,cat, no_grad
from os.path import exists
import os
from NN_resorces.core.network import RAFTGMA
from NN_resorces.core.utils.utils import InputPadder
from NN_resorces.UNET import UNET
from SegmentFlow import calcFow
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from NN_resorces.warp import ForwardWarp

class process_video():#QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    free = pyqtSignal()
    
    def __init__(self, updateBar = None,setMaximum = None,video_dir = '',name = ''):  
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint",default='./weights/gma-chairs.pth')
        parser.add_argument('--model_name', help="define model name", default="GMA")
        parser.add_argument('--num_heads', default=1, type=int,
                            help='number of heads in attention and aggregation')
        parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
        parser.add_argument('--position_and_content', default=False, action='store_true',
                            help='use position and content-wise attention')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        args = parser.parse_args()
        super().__init__()  
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True   
        self.video_dir = video_dir
        self.name = name
        self.updateBar = updateBar  
        self.setMaximum = setMaximum
        self.fwarp = ForwardWarp()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.flow = torch.nn.DataParallel(RAFTGMA(args))
        self.flow.load_state_dict(torch.load(args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
        self.flow.eval()
        self.model = UNET(in_channels=6,channels=[64, 128, 256]).to("cuda")
        if(exists('./weights/gen.pth')):
            self.model.load_state_dict(load('./weights/gen.pth'))
            self.model.eval()
            print('model loaded')
        #self.flow = FlowNet().to(self.device)
        #if(exists('./weights/FlowNetBC.pth')):
        #    self.flow.load_state_dict(load('./weights/FlowNetBC.pth'))
        #    self.flow.eval()

        self.transform = transforms.ToTensor()
        
        
    def process(self):    
        vidcap = cv2.VideoCapture(self.video_dir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setMaximum(frames)
        success,image1 = vidcap.read()
        height, width, c = image1.shape
        if(not exists('./videos')):
          os.mkdir('./videos')  
        T = 2
        video = cv2.VideoWriter('./videos/'+self.name+'.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), fps*T, (width,height))
        iter= 0
        self.progress.emit(iter)
        while success:
            iter=iter+1
            self.addFrameToVideo(image1,video)
            success,image2 = vidcap.read()
            if success:
                newFrame = self.generateFrame(image1,image2,1,T)
                self.addFrameToVideo(newFrame,video)
                #newFrame = self.generateFrame(image1,image2,2,T)
                #self.addFrameToVideo(newFrame,video)
            self.progress.emit(iter)
            image1 = image2
        self.progress.emit(frames)
        print(self.name)
        video.release()
        print('finish')
        self.free.emit()
        self.finished.emit()
    
    def addFrameToVideo(self,frame,video):
        cv2.imwrite("frame.jpg", frame)
        frame = cv2.imread("frame.jpg", cv2.IMREAD_UNCHANGED)
        video.write(frame)
    
    def coords_grid(self,batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)
    
    def warping(self,coords, img):
        N, C, H, W = img.shape
        coords = coords.permute(0, 2, 3, 1)
        xgrid, ygrid = coords.split([1,1], dim=3)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1)
        output = torch.nn.functional.grid_sample(img, grid, align_corners=True)
        return output
    
    def convertTensorToNumpy(self,tensor):
        pool = MaxPool2d(kernel_size=2, stride=2)
        tensor = pool(tensor)
        tensor = pool(tensor)
        tensor = pool(tensor)
        matrix = tensor[0].cpu().permute(1, 2, 0).numpy()
        return matrix
    
    def generateFrame(self, frame1, frame2,t=1,T=2):
        frame1=unsqueeze(self.transform(frame1).to(self.device), dim=0)
        frame2=unsqueeze(self.transform(frame2).to(self.device), dim=0)
        I1o = (frame1 - 0.5) / 0.5
        I2o = (frame2 - 0.5) / 0.5
        padder = InputPadder(I1o.shape)
        I1o, I2o = padder.pad(I1o, I2o)
        f1 = self.convertTensorToNumpy(I1o)
        f2 = self.convertTensorToNumpy(I2o)
        flow1 = calcFow(f1,f2)
        flow2 = calcFow(f2,f1)
        flow1=unsqueeze(self.transform(flow1).to(self.device), dim=0)
        flow2=unsqueeze(self.transform(flow2).to(self.device), dim=0)
        with no_grad():
            _, flow1 = self.flow(I1o, I2o, iters=1, test_mode=True, flow_init=flow1)
            _, flow2 = self.flow(I2o, I1o, iters=1, test_mode=True, flow_init=flow2)
            flow1, flow2 = padder.unpad(flow1)*(t/T), padder.unpad(flow2)*((T-t)/T)
            #flow1,flow2 = self.flow(I1o, I2o,6,False)*(t/T), self.flow(I2o, I1o,6,False)*((T-t)/T)
        _,_,h, w = frame1.shape
        coords = self.coords_grid(1, h, w,self.device)
        #flow1 += coords 
        #flow2 += coords 
        frame1Warped, _ = self.fwarp(frame1,flow1)
        frame2Warped, _ = self.fwarp(frame2,flow2)
        #frame1Warped = self.warping(flow1,frame1)
        #frame2Warped = self.warping(flow2,frame2)
        with no_grad():
            output = self.model(torch.cat([frame1Warped, frame2Warped], dim=1))    
        newFrame = output.cpu().numpy()[0].transpose(1,2,0)*255
        cv2.imwrite("newFrame.jpg", newFrame)
        newFrame = cv2.imread("newFrame.jpg")
        return newFrame, flow1-coords, flow2-coords,frame1Warped.cpu().numpy()[0].transpose(1,2,0),frame2Warped.cpu().numpy()[0].transpose(1,2,0)