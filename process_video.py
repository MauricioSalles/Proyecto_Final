import cv2
import torchvision.transforms as transforms
import torch.cuda as cuda   
import torch
from torch import load,unsqueeze,cat, no_grad
from os.path import exists
from NN_resorces.convNet import convNet
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from NN_resorces.FlowNet import FlowNet

class process_video(QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self, updateBar,setMaximum,video_dir,name):  
        super().__init__()  
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True   
        self.video_dir = video_dir
        self.name = name
        self.updateBar = updateBar  
        self.setMaximum = setMaximum
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.flow = FlowNet().to(self.device)
        self.model = convNet(channels=[32,64,128,256]).to(self.device)
        #if(exists('./weights/convNet2e-5.pth')):
        #    self.model.load_state_dict(load('./weights/convNet2e-5.pth'))
        #    self.model.eval()
        if(exists('./weights/FlowNet.pth')):
            self.flow.load_state_dict(load('./weights/FlowNet.pth'))
            self.flow.eval()
        self.transform = transforms.ToTensor()
        
        
    def process(self):    
        vidcap = cv2.VideoCapture(self.video_dir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setMaximum(frames)
        success,image1 = vidcap.read()
        height, width, c = image1.shape
        video = cv2.VideoWriter(self.name+'.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), fps*2, (width,height))
        iter= 0
        self.progress.emit(iter)
        while success:
            iter=iter+1
            self.addFrameToVideo(image1,video)
            success,image2 = vidcap.read()
            if success:
                newFrame = self.generateFrame(image1,image2)
                self.addFrameToVideo(newFrame,video)
            self.progress.emit(iter)
            image1 = image2
        self.progress.emit(frames)
        print(self.name)
        video.release()
        print('finish')
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
    
    def generateFrame(self, frame1, frame2):
        frame1=unsqueeze(self.transform(frame1).to(self.device), dim=0)
        frame2=unsqueeze(self.transform(frame2).to(self.device), dim=0)
        flow1,flow2 = self.flow(frame1, frame2,5,False), self.flow(frame2, frame1,5,False)
        _,_,h, w = frame1.shape
        coords = self.coords_grid(1, h, w,self.device)
        flow1 += coords 
        flow2 += coords 
        frame1Warped = self.warping(flow1,frame1)
        frame2Warped = self.warping(flow2,frame2)
        with no_grad():
            output = self.model(frame1Warped, frame2Warped)    
        newFrame = output.cpu().numpy()[0].transpose(1,2,0)*255
        cv2.imwrite("newFrame.jpg", newFrame)
        newFrame = cv2.imread("newFrame.jpg")
        return newFrame