# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:19:47 2020

@author: Rohan
"""

from six.moves import urllib
import tensorflow as tf
import zipfile
import os 
from tensorflow.keras.layers import Activation,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
DOWNLOAD_ROOT='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

local_file='C:/Users/Rohan/Downloads/cats_and_dogs_filtered.zip'
zipfile=zipfile.ZipFile(local_file,'r')
zipfile.extractall('/tmp')
zipfile.close()

base_dir='/tmp/cats_and_dogs_filtered'
train_dir=os.path.join(base_dir,'train')
val_dir=os.path.join(base_dir,'validation')

train_cat_dir=os.path.join(train_dir,'cats')
train_dog_dir=os.path.join(train_dir,'dogs')

'''n_cols=4
n_rows=4
img_index=0
train_cat_fname=os.listdir(train_cat_dir)
train_dog_fname=os.listdir(train_dog_dir)

img=plt.gcf()
img=img.set_size_inches(3,4)
temp_cat_fname=[os.path.join(train_cat_dir,fname) for fname in train_cat_fname[img_index:img_index+8]]
temp_dog_fname=[os.path.join(train_dog_dir,fname) for fname in train_dog_fname[img_index:img_index+8]]

for i,path in enumerate(temp_cat_fname+temp_dog_fname):
    img.subplots(n_rows,n_col,i+1)
    image=pimg.imread(path)
    plt.imshow(image)
    '''
    
imagegenerator=ImageDataGenerator()
train_img=imagegenerator.flow_from_directory(train_dir,class_mode='binary',batch_size=128,target_size=(300,300))
val_img=imagegenerator.flow_from_directory(val_dir)

fig,axarr=plt.subplots(3,4)

model=Sequential([Conv2D(16,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Conv2D(32,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Conv2D(64,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Flatten(),Dense(512,activation='relu'),Dense(1,activation='sigmoid')])
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(train_img,epochs=10,verbose=2)
'''model=Sequential([Conv2D(64,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Conv2D(32,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Conv2D(16,(3,3),activation='relu'),MaxPooling2D(2,2),BatchNormalization(),Flatten(),Dense(512,activation='relu'),Dense(1,activation='sigmoid')])
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(train_img, epochs=10, steps_per_epoch=5,verbose=2)


layer_outputs=[layer.output for layer in model.layers[1:]]
model1=tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)
successive_feature_maps=model1.predict(doggy)

*******************************************************
Important 
-put input size in the first layer or else the outputs cannot be gathereed
-also mention class=binay in imageDataGenerator
'''

layer_name=[layer.name for layer in model.layers]
for layer,feature in zip(layer_name,successive_feature_maps):
    n_features=feature_map.shape[-1:]
    n_cols=feature_map.shape[1]
    digspace=np.zeros((n_cols,n_cols*n_features))
    for i in range(n_features):
        x=feature[0,:,:,i]
        x=(x-x.mean())/x.std()
        x=x*64
        x=x+128
        x=np.clip(x,0,255).astype('uint8')
        digspace[:,i*size:(i+1)*size]=x
        scale=20./n_features
        plt.figure(figsize=(scale*n_features,scale))
        plt.imshow(digspace,aspect='auto',cmap='virdis')