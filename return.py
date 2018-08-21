train_path = 'data/train/'
train1_path = 'data/train1/'
cut_path = 'data/cut/'
import os, cv2
import shutil

labels = os.listdir(train_path)
#labels = ['1']

for label in labels:
    t_imgs = os.listdir(train_path + label)
    c_imgs = os.listdir(cut_path + label)
    cc = []
    for cimg in c_imgs:
        a = cimg.replace('_0','')
        cc.append(a)
    for img in t_imgs:
        a=0
        for cimg in cc:
            if(cimg==img):
                a=1
                break
        if(a==0):
            shutil.copy(train_path + label + "/" + img, cut_path + label + "/" + img)

