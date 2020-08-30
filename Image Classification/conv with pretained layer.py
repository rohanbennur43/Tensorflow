# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:33:32 2020

@author: Rohan
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from tensorflow.keras.layers import Activation,Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout


cats_and_dogs='C:/Users/Rohan/Downloads/cats_and_dogs_filtered.zip'
local_model_path='C:/Users/Rohan/Downloads/PROJ/h5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

zipfile=zipfile.ZipFile(cats_and_dogs,'r')
zipfile.extractall('/tmp')
zipfile.close()

base_dir='/tmp/cats_and_dogs_filtered'
train_dir=os.path.join(base_dir,'train')
val_dir=os.path.join(base_dir,'validation')

train_cat_dir=os.path.join(train_dir,'cats')
train_dog_dir=os.path.join(train_dir,'dogs')

img_generator=ImageDataGenerator()
images_dir=img_generator.flow_from_directory(train_dir,target_size=(150,150),class_mode='binary')
val_images=img_generator.flow_from_directory(val_dir,target_size=(150,150),class_mode='binary')

'''pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_model_path)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

print('1')

x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
print('2')
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           
print('3')
model = tf.keras.models.Model( pre_trained_model.input, x) 
print('4')
model.compile(optimizer = 'Adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
************************************************
error noticed at accuracy do not use Accuracy'''

model=InceptionV3(input_shape=(150,150,3),include_top=False,weights=None)
model.load_weights(local_model_path)

for layers in model.layers:
    layers.trainable=False
print('1')
layer=model.get_layer('mixed7')
last_output=layer.output
print('2')
#x=tf.keras.layers.Lambda(lambda x:tf.nn.max_pool(x,ksize=(1,1,1,3),strides=(1,1,1,3),padding='VALID'))(last_output)

x=Flatten()(last_output)
x=Dropout(rate=0.5)(x)
x=Dense(512,activation='relu')(x)
x=Dropout(rate=0.5)(x)
x=Dense(1,activation='sigmoid')(x)
model_fin=tf.keras.models.Model(inputs=model.input,outputs=x)
print('3')


model_fin.compile(optimizer = 'Adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
print('4')
model_fin.fit_generator(images_dir,epochs=5,verbose=2,validation_data=val_images)
print('5')