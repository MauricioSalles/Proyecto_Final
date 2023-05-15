import os
from warp import Warp
from cv2 import imread, imwrite, IMREAD_UNCHANGED

def frameList(dir , Warp):
        for directories in os.listdir(dir):
            images =  os.listdir(dir + '\\' + directories)
            images.sort()
            directory_frames = []
            for i in range(len(images)-2):
                img1 = dir + '\\' +directories +'\\' + images[i]
                img2 = dir + '\\' +directories +'\\' + images[i+1]
                img3 =dir + '\\' +directories +'\\' +  images[i+2]
                input = (img1,img3)
                directory_frames.append(input)
            
            if not os.path.exists(dir + '\\' +directories + '\\'+ "warped"):
                os.mkdir(dir + '\\' +directories + '\\'+ "warped")
                os.mkdir(dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f1")
                os.mkdir(dir + '\\' +directories + '\\'+"warped"+ '\\' + "f2")
                idx = 0
                for (img1 , img2) in directory_frames:
                    image1  = imread(img1, IMREAD_UNCHANGED )
                    image2  = imread(img2, IMREAD_UNCHANGED )
                    f1 = Warp.warpImage(image1,image1, image2)
                    f2 = Warp.warpImage(image2,image2, image1)
                    f1name = dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1/frame%d.jpg" % (idx+10000)
                    f2name = dir + '\\'+directories + '\\' +"warped"+ '\\' + "f2/frame%d.jpg" % (idx+10000)
                    imwrite(f1name, f1)
                    imwrite(f2name, f2)
                    idx += 1
                    
dire = './dataset2'
warp = Warp()
frameList(dire, warp)
