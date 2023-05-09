import os
from cv2 import imread as open

class FramesDataset():
    
    def __init__(self, dir,transform = None):
        self.transform = transform
        self.dir = dir
        print(os.getcwd())
        self.frames = self.frameList()
        
    def __len__(self):
        print(len(self.frames))
        return len(self.frames)
    
    def __getitem__(self,index):
        (img1,img2, img3) =  self.frames[index]
        img1 = open(img1)
        img2 = open(img2)
        img3 = open(img3)
        return (img1,img2, img3)

    def frameList(self):
        directory_frames = []
        for directories in os.listdir(self.dir):
            images =  os.listdir(self.dir + '\\' + directories)
            images.sort()
            for i in range(len(images)-3):
                img1 = self.dir + '\\' +directories +'\\' + images[i]
                img2 = self.dir + '\\' +directories +'\\' + images[i+1]
                img3 =self.dir + '\\' +directories +'\\' +  images[i+2]
                image = (img1,img2,img3)
                directory_frames.append(image)  
        return directory_frames 
    