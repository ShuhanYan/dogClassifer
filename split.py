import os
import shutil
import random


for dir in os.listdir("data/test1"):
    print(dir)
    i=0
    for filename in os.listdir("data/test1/"+dir):
        if os.path.exists("data/val/" + dir) == False:
            os.mkdir("data/val/"+dir)
        if i>10:
            continue
        i+=1
        shutil.copy("data/test1/"+dir+"/"+filename,"data/val/"+dir+"/"+filename)
