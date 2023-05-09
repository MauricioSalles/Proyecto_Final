import cv2
import os

dataset = "dataset"

def ob_img(video_dir, name):
    vidcap = cv2.VideoCapture(video_dir)
    success,image = vidcap.read()
    count = 100000
    if not os.path.exists(f"./{dataset}"):
        os.mkdir(f"./{dataset}")
    if os.path.exists(f"./{dataset}/{name}"):
        return
    os.mkdir(f"./{dataset}/{name}")
    os.chdir(f"./{dataset}/{name}")
    while success:
        imgname = "%d.jpg" % count
        cv2.imwrite(imgname, image)
        success,image = vidcap.read()
        count += 1
    os.chdir(f"../..")

os.chdir("..")
directory = os.getcwd()+"/{dataset}"
for video in os.listdir("video"):
    dir = os.getcwd()+r"\video"+ "\\" + video
    name = video.split(".")[0]
    ob_img(dir,name)
