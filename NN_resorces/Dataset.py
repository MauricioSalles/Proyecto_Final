import os
import torch.cuda as cuda
from torch.utils.data import Dataset
from PIL.Image import open

class FramesDataset(Dataset):
    
    def __init__(self, dir, transform = None):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.transform = transform
        self.dir = dir #'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset'
        self.frames = self.frameList()
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        ((frame1,frame3),frame2,(img1, img3)) =  self.frames[index]
        img1 = open(img1)
        img3 = open(img3)
        frame1 = open(frame1)
        frame2 = open(frame2)
        frame3 = open(frame3)
        if self.transform is None:
            return ((frame1, frame3), frame2, (img1,img3))
        img1 = self.transform(img1)
        img3 = self.transform(img3)
        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)
        return ((frame1,frame3), frame2, (img1, img3))

    def frameList(self):
        frames = []
        for directories in os.listdir(self.dir):
            F1frames = []
            F2frames = []
            output = []
            images =  os.listdir(self.dir + '\\' + directories)
            images.sort()
            directory_frames = []
            for i in range(len(images)-3):
                img1 = self.dir + '\\' +directories +'\\' + images[i]
                img2 = self.dir + '\\' +directories +'\\' + images[i+1]
                img3 =self.dir + '\\' +directories +'\\' +  images[i+2]
                input = (img1,img3)
                output.append(img2)
                directory_frames.append(input)
            
            if not os.path.exists(self.dir + '\\' +directories + '\\'+ "warped"):
                print("directory doesnt exists")
            else:
                F1frames = os.listdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f1")                
                F2frames = os.listdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f2")     
                for idx in range(len(F1frames)):
                    f1 = F1frames[idx]
                    f2 = F2frames[idx]
                    f1 = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1\\"+f1
                    f2 = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f2\\"+f2 
                    frames.append(((f1,f2), output[idx], directory_frames[idx]))     
        return frames 
    