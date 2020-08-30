# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:18:40 2020

@author: Rohan
"""
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,Activation,MaxPooling2D,Flatten,BatchNormalization
from tensorflow.keras.models import Sequential
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

local_zip='C:/Users/Rohan/Downloads/horse-or-human.zip'
zip_ref=zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()

train_horse_dir=os.path.join('C:/Users/Rohan/Downloads/horse-or-human/horse')
train_human_dir=os.path.join('C:/Users/Rohan/Downloads/horse-or-human/human')

image_generator=ImageDataGenerator()
train_data=image_generator.flow_from_directory('/tmp/horse-or-human',target_size=(300,300),batch_size=128,class_mode='binary')

model=Sequential([Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),MaxPooling2D(2,2),BatchNormalization(),Conv2D(32,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Conv2D(64,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Flatten(),Dense(512,activation='relu'),Dense(1,activation='sigmoid')])
x=tf.keras.layers.Lambda(lambda x:tf.nn.max_pool(x,ksize=(1,1,1,3),strides=(1,1,1,3),padding='valid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(train_data,epochs=1,verbose=2)


prajwal=Image.load_img(path,target_size)
prajwal=np.expand_dims(prajwal,axis=0)
'''
def fetch_housing_data(housing_url=HOUSING_URL)
    urllib.request.urlretrieve(housing_url,tgz_path)
    zipfile=zipfile.ZipFile(tgz_path,'r')
    zipfile.extractallall()
    zipfile.close()'''