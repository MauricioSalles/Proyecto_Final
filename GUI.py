import sys
import torchvision.transforms as transforms
import torch.cuda as cuda   
from process_video import process_video
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QFileDialog, QPushButton, QLabel, QProgressBar
from PyQt5.QtCore import QThread

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
        self.transform = transforms.ToTensor()
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
        self.thread = QThread()
        self.process_video = process_video(updateBar=self.updateBar, setMaximum=self.progressBar.setMaximum, video_dir=self.videoDir, name=self.textbox.text())
        self.process_video.moveToThread(self.thread)
        self.thread.started.connect(self.process_video.process)
        self.process_video.progress.connect(self.updateBar)
        self.process_video.finished.connect(self.thread.quit)
        self.process_video.finished.connect(self.process_video.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        print('finish')
    
    def updateBar(self,int):
        self.progressBar.setValue(int)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
