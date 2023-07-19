import sys
from torch import load,unsqueeze,cat, no_grad
from process_video import process_video
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel, QProgressBar
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import cv2
from torch import load,unsqueeze,cat, no_grad
from os.path import exists
from NN_resorces.convNet import convNet
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel, QProgressBar
from PyQt5.QtGui import QIcon
from NN_resorces.refine_flow import refine_flow
import torchvision.transforms as transforms
import torch.cuda as cuda   

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Interpolacion'
        self.left = 200
        self.top = 200
        self.width = 500
        self.height = 200
        self.videoDir = ""
        self.main()
        
    def main(self):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = QLabel('video',self)
        self.label.setGeometry(10,160,200,25)
        self.buttonSearch = QPushButton('buscar archivo', self)
        self.buttonSearch.setGeometry(10,20,240,25)
        self.buttonSearch.clicked.connect(self.openFileSearch)
        self.buttonVideo = QPushButton('Procesar video', self)
        self.buttonVideo.setGeometry(250,20,240,25)
        self.buttonVideo.clicked.connect(self.processVideo)
        self.label2 = QLabel('Nombre del video',self)
        self.label2.setGeometry(10,50,200,30)
        self.textbox = QLineEdit(self)
        self.textbox.setGeometry(10,80,480,30)
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setGeometry(10,120,480,30)
        self.model = convNet().to(self.device)
        if(exists("./NN_resorces/weights.pth")):
            self.model.load_state_dict(load('./NN_resorces/weights.pth'))
            self.model.eval()
        self.transform = transforms.ToTensor()
        self.flow = refine_flow()
        self.show()
        
    def openFileSearch(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","Videos (*.mp4)", options=options)
        if files:
            file = files[0]
            self.videoDir = file
            name = file.split("/")
            self.label.setText(name[-1])
            print(files[0])
    
    def processVideo(self):
        if self.videoDir == "":
            print ("no hay direccion del video")
            return
        #self.process()
        self.thread = QThread()
        self.process_video = process_video(updateBar=self.updateBar, setMaximum=self.progressBar.setMaximum, video_dir=self.videoDir, name=self.textbox.text())
        #self.process_video.process()
        self.process_video.moveToThread(self.thread)
        self.thread.started.connect(self.process_video.process)
        self.thread.start()
        print('finish')
    
    def updateBar(self,int):
        self.progressBar.setValue(int)
        
    def process(self):    
        newVideo = []
        vidcap = cv2.VideoCapture(self.videoDir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progressBar.setMaximum(frames)
        success,image1 = vidcap.read()
        iter= 0
        self.progressBar.setValue(iter)
        while success:
            iter=iter+1
            success,image2 = vidcap.read()
            if success:
                newFrame = self.generateFrame(image1,image2)
                newVideo.append(newFrame)
            self.progressBar.setValue(iter)
            cv2.imwrite("frame.jpg", image1)
            frame = cv2.imread("frame.jpg", cv2.IMREAD_UNCHANGED)
            newVideo.append(frame)
            image1 = image2
        self.progressBar.setValue(frames)
        height, width, c = newVideo[-1].shape
        video = cv2.VideoWriter(self.textbox.text()+'.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), fps*2, (width,height))
        for i in range(len(newVideo)):
            video.write(newVideo[i])
        video.release()
        
    def generateFrame(self, frame1, frame2):
        flow1 = self.flow.calcFlow(frame1, frame2)/2
        flow2 = self.flow.calcFlow(frame2, frame1)/2
        h, w = flow1.shape[:2]
        flow1[:,:,0] += np.arange(w)
        flow1[:,:,1] += np.arange(h)[:,np.newaxis]
        flow2[:,:,0] += np.arange(w)
        flow2[:,:,1] += np.arange(h)[:,np.newaxis]
        frame1Warped = cv2.remap(frame1, flow1,None, cv2.INTER_LINEAR)
        frame2Warped = cv2.remap(frame2, flow2,None, cv2.INTER_LINEAR)
        f1w = frame1Warped
        frame1Warped = self.transform(frame1Warped)
        frame2Warped = self.transform(frame2Warped)  
        with no_grad():
            output = self.model(frame1Warped.to(self.device), frame2Warped.to(self.device))    
        newFrame = output.cpu().numpy()[0].transpose(1,2,0)*255
        cv2.imwrite("newFrame.jpg", f1w)
        newFrame = cv2.imread("newFrame.jpg")
        return newFrame


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
