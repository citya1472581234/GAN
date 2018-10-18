# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:53:24 2018

@author: USER
"""

import os 
import shutil
image_dir = "D:/Downloads/lfw/lfw/"
target = "C:/Users/USER/Desktop/face_3000/"
files= os.listdir(image_dir)

count=0;
for file in files:
    image = image_dir+file
    image_file = os.listdir(image)
    for img in image_file:
        result = image+'/'+img
        shutil.copy(result,target)
        count +=1
        if count ==3000:
            break


