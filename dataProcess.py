import numpy as np
import os
import shutil
import h5py
import random
import pandas as pd
Dict={}

with open('data/label_train.txt','r') as f:
    data = f.readlines()
    for line in data:
        b = line.split(' ')
        if os.path.exists('data/train/'+b[0]):
            if os.path.exists('data/train/'+b[1])==False:
                os.mkdir("data/train/"+b[1])
            shutil.move("data/train/"+b[0],'data/train/'+b[0]+'/'+b[1])
