# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:42:16 2020

@author: Rohan
"""
import tensorflow
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense
i,j,k=75,15,10
images,info=tfds.load('tf_flowers',as_supervised=True,with_info=True)
train_set=tfds.load('tf_flowers',as_supervised=True,split='train[:75%]')
valid_set=tfds.load('tf_flowers',as_supervised=True,split='train[75%:90%]')
test_set=tfds.load('tf_flowers',as_supervised=True,split='train[90%:]')
n_classes=5
def preprocess(images,labels):
    resized_image=tf.image.resize(images,[274,274])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image,labels

train_set=train_set.map(preprocess)
train_set=train_set.batch(32)


  

base_model = keras.applications.xception.Xception(weights="imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.Trainable=False
'''base_model=tensorflow.keras.applications.xception.Xception(weights='imagenet',include_top='False')
avg=tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
dense=Dense(512,activation='relu')(avg)
dense=Dense(5,avtivation='softmax')(dense)'''

fin_model=tensorflow.keras.models.Model(inputs=base_model.input,outputs=output)


fin_model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
fin_model.fit(train_set,epochs=5,verbose=2,steps_per_epoch=100)
'''************************************
make it into batches to add another dimension'''