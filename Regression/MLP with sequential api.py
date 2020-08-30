# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:09:35 2020

@author: Rohan
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing=fetch_california_housing()
X_train_full,X_test,y_train_full,y_test=train_test_split(housing.data,housing.target)
X_train,X_valid,y_train,y_valid=train_test_split(X_train_full,y_train_full)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_valid_scaled=scaler.transform(X_valid)

model=keras.Sequential([Dense(30,activation='relu',input_shape=X_train.shape[1:]),Dense(1)])
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
history=model.fit(X_train_scaled,y_train,epochs=30)
