# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:22:43 2022

@author: A_chulab
"""

import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import random

def generate_numbers():
    while True:
        numbers = [random.randint(30, 330) for R in range(4)]
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                if numbers[i] + numbers[j] == 360:
                    valid = False
                    break
        if valid:
            return numbers
data_raw=[]
data_mask=[]
rawpath = input('INSERT THE DATA PATH :\n')
maskpath = input('INSERT THE LABEL PATH :\n')
savedatapath = input('INSERT THE SAVE_DATA PATH :\n')
savemaskpath = input('INSERT THE SAVE_LABEL PATH :\n')
number = input('number :\n')

print('reading image')

rawfilelist = sorted(os.listdir(rawpath))
maskfilelist = sorted(os.listdir(maskpath))

for rawfile in tqdm(rawfilelist):
    img = cv2.imread(rawpath+'\\'+rawfile,2)
    
    data_raw.append(img)


for maskfile in tqdm(maskfilelist):
    img_mask = cv2.imread(maskpath+'\\'+maskfile,2)
    #mask
    img_mask = img_mask.astype('uint8')
    data_mask.append(img_mask)
   # data_mask.append(img_mask)
    '''
    #erode & opening
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img_mask,kernel)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(erosion,cv2.MORPH_OPEN,kernel) 
    data_mask.append(opening)
    '''



#切成適合模型訓練大小
print('cut')

size = 256
ilength = data_raw[0].shape[0]
jlength = data_raw[0].shape[1]

train_raw_list=[] 
train_mask_list=[]

for x in tqdm(range(len(data_raw))):
    for i in range(0,ilength-size,128):
        for j in range(0,jlength-size,128):
            train_raw_data = np.zeros((size,size),dtype='uint16')
            train_mask_data = np.zeros((size,size),dtype='uint8')

            train_raw_data = data_raw[x][i:i+size,j:j+size]
            train_mask_data = data_mask[x][i:i+size,j:j+size]
            
           # train_raw_data = cv2.resize(train_raw_data,(128,128))
           # train_mask_data = cv2.resize(train_mask_data,(128,128))
            
            train_raw_list.append(train_raw_data)
            train_mask_list.append(train_mask_data)


#%%(data augmentation)

idx = len(train_raw_list)

## flipping vertical

Flip_data_vertically = np.zeros([size,size],dtype = 'uint16')
Flip_label_vertically = np.zeros([size,size],dtype = 'uint8')


for i in range(idx):
    #index = np.arange(size) #0~255
    #x = size-index -1 
   # Flip_data_vertically =train_raw_list[i][x,:]
    
    Flip_data_vertically = cv2.flip(train_raw_list[i], 0)#my
    train_raw_list.append(Flip_data_vertically)
    
    #Flip_label_vertically = train_mask_list[i][x,:]
    
    Flip_label_vertically = cv2.flip(train_mask_list[i], 0)#my
    train_mask_list.append(Flip_label_vertically)
  

## flipping horizontal

Flip_data_horizontal = np.zeros([size,size],dtype = 'uint16')
Flip_label_horizontal = np.zeros([size,size],dtype = 'uint8')

'''
for i in range(idx):
    #index = np.arange(size) #0~255
    #y = size-index -1 
    #Flip_data_horizontal =train_raw_list[i][:,y]
    
    Flip_data_horizontal = cv2.flip(train_raw_list[i], 1)
    train_raw_list.append(Flip_data_horizontal)
    
    #Flip_label_horizontal = train_mask_list[i][:,y]
    Flip_label_horizontal = cv2.flip(train_mask_list[i], 1)
    train_mask_list.append(Flip_label_horizontal)
'''
numbers = generate_numbers()


## rotate 90

Rotate_data = np.zeros([size,size],dtype = 'uint16')
Rotate_label = np.zeros([size,size],dtype = 'uint8')



## rotate 180

Rotate_data = np.zeros([size,size],dtype = 'uint16')
Rotate_label = np.zeros([size,size],dtype = 'uint8')

for i in range(idx):
    (h,w) = train_raw_list[0].shape
    M = cv2.getRotationMatrix2D((h/2,w/2), 90, 1) # (center,angle,scale)
    rotated = cv2.warpAffine(train_raw_list[i], M, (w, h))
    train_raw_list.append(rotated)      
     
    rotated_mask = cv2.warpAffine(train_mask_list[i], M, (w, h))
    train_mask_list.append(rotated_mask)

'''
for i in range(idx):
    (h,w) = train_raw_list[0].shape
    M = cv2.getRotationMatrix2D((h/2,w/2), 180, 1) # (center,angle,scale)
    rotated = cv2.warpAffine(train_raw_list[i], M, (w, h))
    train_raw_list.append(rotated)      
     
    rotated_mask = cv2.warpAffine(train_mask_list[i], M, (w, h))
    train_mask_list.append(rotated_mask)
'''
#for i in range(idx):
#    (h,w) = train_raw_list[0].shape
#    M = cv2.getRotationMatrix2D((h/2,w/2), numbers[3], 1) # (center,angle,scale)
#    rotated = cv2.warpAffine(train_raw_list[i], M, (w, h))
#    train_raw_list.append(rotated)      
     
#    rotated_mask = cv2.warpAffine(train_mask_list[i], M, (w, h))
#    train_mask_list.append(rotated_mask)    

## rotate 270


Rotate_data = np.zeros([size,size],dtype = 'uint16')
Rotate_label = np.zeros([size,size],dtype = 'uint8')

'''

for i in range(idx):
    (h,w) = train_raw_list[0].shape
    M = cv2.getRotationMatrix2D((h/2,w/2), 270, 1) # (center,angle,scale)
    rotated = cv2.warpAffine(train_raw_list[i], M, (w, h))
    train_raw_list.append(rotated)      
     
    rotated_mask = cv2.warpAffine(train_mask_list[i], M, (w, h))
    train_mask_list.append(rotated_mask)


'''
#train_mask_list = train_mask_list+back_mask

#存圖

print('saving image')
#os.makedirs(savepath+'/'+'train_raw'+number)

for l in tqdm(range(len(train_raw_list))):
    Image.fromarray(train_raw_list[l]).save(savedatapath+'\\'+str(l)+'_'+ number +'.tif')
    Image.fromarray(train_mask_list[l]).save(savemaskpath+ '\\' + str(l)+'_'+ number +'.tif')  
