# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:45:51 2023

@author: ChuLab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:20:41 2023

@author: ChuLab
"""

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from training_functions import *



# Reading Raw Data
raw_data_path = r''
raw_mask_path = r''

rawfilelist = [os.path.join(raw_data_path, rawfile) for rawfile in sorted(os.listdir(raw_data_path))]
maskfilelist = [os.path.join(raw_mask_path, maskfile) for maskfile in sorted(os.listdir(raw_mask_path))]


data_raw = [cv2.imread(rawfile, cv2.IMREAD_UNCHANGED) for rawfile in tqdm(rawfilelist)]
data_mask = [cv2.imread(maskfile, cv2.IMREAD_UNCHANGED) for maskfile in tqdm(maskfilelist)]


augmented_data, augmented_mask =  bright_add(data_raw[0:16], data_mask[0:16],100,50)
training_raw = augmented_data
training_mask = augmented_mask

augmented_data, augmented_mask =  bright_add(data_raw[16:32], data_mask[16:32],150,150)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))


augmented_data, augmented_mask =  bright_add(data_raw[32:48], data_mask[32:48],200,50)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))

augmented_data, augmented_mask =  bright_multiply(data_raw[48:64], data_mask[48:64],3,50)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))

augmented_data, augmented_mask =  bright_multiply(data_raw[64:80], data_mask[64:80],2,50)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))

augmented_data, augmented_mask =  bright_multiply(data_raw[80:100], data_mask[80:100],1,50)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))

'''
augmented_data, augmented_mask =  bright_multiply(data_raw[100:200], data_mask[100:200],4,32)
training_raw = np.concatenate((training_raw, augmented_data))
training_mask = np.concatenate((training_mask, augmented_mask))

'''

training_raw = np.array(training_raw)





training_mask = np.array(training_mask)





training_data_path = r''
training_mask_path = r''
os.makedirs(training_data_path, exist_ok = True)
os.makedirs(training_mask_path, exist_ok = True)
for i in tqdm(range(len(training_raw))):
    Image.fromarray(training_raw[i]).save(training_data_path + f"\\{i:0>{6}}.tif")
    Image.fromarray(training_mask[i]).save(training_mask_path + f"\\{i:0>{6}}.tif")




