import cv2
import os
dir = r'./pruebas'
for path in os.listdir("pruebas"):
    name = path.split('.')[0]
    read_path = os.getcwd()+r"\pruebas"+ "\\" + path
    save_path_full = os.getcwd()+r"\slowed"+ "\\" + path
    save_path_name = os.getcwd()+r"\slowed"+ "\\" + name
    #if(True):
    if(not os.path.isfile(save_path_full)):
        vidcap = cv2.VideoCapture(read_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image1 = vidcap.read()
        height, width, c = image1.shape
        video = cv2.VideoWriter(save_path_name + '_slowed'+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps/2, (width,height))
        iter= 0
        while success:
            iter=iter+1
            if (iter%2 == 0):
                cv2.imwrite("temp.jpg", image1)
                frame = cv2.imread("temp.jpg", cv2.IMREAD_UNCHANGED)
                video.write(frame)
            success,image2 = vidcap.read()
            image1 = image2
        video.release()