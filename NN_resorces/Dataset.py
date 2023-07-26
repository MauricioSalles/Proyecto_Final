import os
import torch.cuda as cuda
from torch.utils.data import Dataset
from cv2 import imread as open

class FramesDataset(Dataset):
    
    def __init__(self, dir, transform = None):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.transform = transform
        self.dir = dir #r'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset'
        self.slash = '\\'
        self.frames = self.frameList()
        #esto debido a que al entrenar en google colab se rrequiere otro tipo de slash
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        (frame1,frame2,frame3) =  self.frames[index]
        frame1 = open(frame1)
        frame2 = open(frame2)
        frame3 = open(frame3)
        if self.transform is None:
            return (frame1,frame2,frame3)
        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)
        return (frame1,frame2,frame3)

    def frameList(self):
        frames = []
        for directories in os.listdir(self.dir):
            F1frames = []
            F2frames = []
            output = []
            images =  os.listdir(self.dir + self.slash + directories)
            images.sort()
            for i in range(len(images)-3):
                img2 = self.dir + self.slash +directories +self.slash + images[i+1]
                output.append(img2)
            
            if not os.path.exists(self.dir + self.slash +directories + self.slash+ "warped"):
                print("directory doesnt exists")
            else:
                F1frames = os.listdir(self.dir + self.slash +directories + self.slash+"warped"+ self.slash  + "f1")                
                F2frames = os.listdir(self.dir + self.slash +directories + self.slash+"warped"+ self.slash  + "f2")     
                for idx in range(len(F1frames)):
                    f1 = F1frames[idx]
                    f2 = F2frames[idx]
                    f1 = self.dir + self.slash+directories + self.slash +"warped"+ self.slash + "f1\\"+f1
                    f2 = self.dir + self.slash+directories + self.slash +"warped"+ self.slash + "f2\\"+f2 
                    frames.append((f1,output[idx],f2))     
        return frames 
    