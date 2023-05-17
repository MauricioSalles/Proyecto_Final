import os
import shutil

dir = './dataset2'
for directories in os.listdir(dir):
        if os.path.exists(dir + '\\' +directories + '\\'+ "warped"):
                shutil.rmtree(dir + '\\' +directories + '\\'+ "warped")

                    
