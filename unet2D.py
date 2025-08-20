# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:08:26 2022

@author: A_chulab
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
import random
import warnings
import time 
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from PIL import ImageFile

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

from evaluate import *


#%% (data pre-processing ) 
#tf.compat.v1.enable_eager_execution()

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1


datapath = input('INSERT THE DATA PATH :\n')
savemodelpath = input('INSERT WHERE THE MODEL SAVED PATH :\n')
name = input("INSERT THE EXPERIMENT NAME:\n")
trainrawpath = datapath+'/'+'train_data/'
trainmaskpath = datapath+'/'+'train_label/'
#seed = 42
#random.seed = seed
np.random.seed = 42
start_time=time.time()
image_ids = next(os.walk(trainrawpath))[2]
 
X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS),dtype=np.float16)
Y = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
Y.shape
for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    path = trainrawpath + id_
    img = imread(path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = np.expand_dims(img, axis=2)
    X[n] = img

mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
mask_file=next(os.walk(trainmaskpath))[2]
for n in range(len(mask_file)):
    mask_path = trainmaskpath + mask_file[n]
    mask = imread(mask_path,-1)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    mask = np.expand_dims(mask, axis=2)
    Y[n] = mask


#shuffle
index=[i for i in range(len(X))]
np.random.shuffle(index)
X = X[index]
Y = Y[index]
print("---------shuffle-------------")

x_train=X
y_train=Y

#plt.imshow(np.squeeze(x_train[8000]),cmap='gray')
#plt.show()

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))  # 修改这里
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))  # 修改这里
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dic = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    dice = 1 - dic


    return dice
def iou_metric(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return intersection / (union + tf.keras.backend.epsilon())
 

#%% (Architecture)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x / 1) (inputs)

c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = BatchNormalization()(c1)
c1 = Dropout(0.5)(c1)
c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = BatchNormalization()(c2)
c2 = Dropout(0.5)(c2)
c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = BatchNormalization()(c3)
c3 = Dropout(0.5)(c3)
c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = BatchNormalization()(c4)
c4 = Dropout(0.5)(c4)
c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = BatchNormalization()(c5)
c5 = Dropout(0.5)(c5)
c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(0.5)(c6)
c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(0.5)(c7)
c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(0.5)(c8)
c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = BatchNormalization()(c9)
c9 = Dropout(0.5)(c9)
c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
c9 = BatchNormalization()(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)


model = Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer = 'adam', loss=dice_loss, metrics=['accuracy'])
model.summary()



#%% (Training)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(savemodelpath+"\\"+str(name)+".h5", verbose=1,save_best_only=(True))
results = model.fit(x_train, y_train, validation_split=0.2, batch_size=12, epochs=60, 
                    callbacks=[earlystopper, checkpointer]) 

#%% (Evaluation)
evaluation_df = pd.DataFrame({"accuracy":[],"val_accuracy":[]})
evaluation_df.loc[:,"accuracy"]=results.history['accuracy']
evaluation_df.loc[:,"val_accuracy"]=results.history['val_accuracy']
evaluation_df.to_excel(savemodelpath + "\\"+str(name)+".xlsx", index = False)

plt.plot(results.history['accuracy'], label='accuracy')
plt.plot(results.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

#plt.ylim(0.99,1)
plt.legend(loc='lower right')
plt.tight_layout() #避免圖標跑掉
#plt.show()
plt.savefig(savemodelpath +"\\" +str(name) +".png")
end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")
