# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:57:56 2020

@author: Rohan
"""

import tensorflow1
from tensorflow import keras
from tensorflow.keras.layers import Concatenate,Dense,Activation,Input
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing=fetch_california_housing()
X_train_full,X_test,y_train_full,y_test=train_test_split(housing.data,housing.target)
X_train,X_valid,y_train,y_valid=train_test_split(X_train_full,y_train_full)

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]


'''input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],outputs=[output, aux_output])
'''

input1 = keras.layers.Input(shape=[5])
input2=keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input1)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input2, hidden2])
output1 = keras.layers.Dense(1)(concat)
output2=Dense(1,activation='softmax')(hidden2)
model = keras.models.Model(inputs=[input1,input2], outputs=[output1,output2])





'''input = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input], outputs=[output])'''


model.compile(loss=['mse','mse'],loss_weights=[0.9,0.1],optimizer='sgd',metrics=['accuracy'])
history=model.fit(x=[X_train_A,X_train_B],y=[y_train,y_train],epochs=5)
