import sys
import PyQt5
import os
import cv2
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel
from PyQt5.QtGui import QIcon

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Interpolacion'
        self.left = 100
        self.top = 100
        self.width = 500 
        self.height = 400
        self.videoDir = ""
        self.main()
        
    def main(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = QLabel('video',self)
        self.label.setGeometry(10,350,200,10)
        self.buttonSearch = QPushButton('buscar archivo', self)
        self.buttonSearch.move(10,10)
        self.buttonSearch.clicked.connect(self.openFileSearch)
        self.buttonVideo = QPushButton('Procesar video', self)
        self.buttonVideo.move(10,35)
        self.buttonVideo.clicked.connect(self.procesarVideo)
        self.show()
        
    def openFileSearch(self):
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNamesDialog()
    
    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","Videos (*.mp4)", options=options)
        if files:
            file = files[0]
            self.videoDir = file
            name = file.split("/")
            self.label.setText(name[-1])
            print(files[0])
    
    def procesarVideo(self):
        newVideo = []
        video_dir = self.videoDir
        if self.videoDir == "":
            print ("no hay direccion del video")
            return
        vidcap = cv2.VideoCapture(video_dir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image = vidcap.read()
        iteracion= 0
        while success:
            iteracion=iteracion+1
            cv2.imwrite("frame.jpg", image)
            frame = cv2.imread("frame.jpg", cv2.IMREAD_UNCHANGED)
            newVideo.append(frame)
            success,image = vidcap.read()
        height, width, c = newVideo[-1].shape
        video = cv2.VideoWriter('generated.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), fps*2, (width,height))
        for i in range(len(newVideo)):
             video.write(newVideo[i])
        video.release()
        print('finish')
    



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
