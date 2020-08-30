# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:30:37 2020

@author: Rohan
"""
import tensorflow
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Embedding
from tensorflow.keras.models import Sequential
import numpy as np

download_url='https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt'
filepath=get_file('laurencepoetry.txt',download_url)

with open(filepath) as f:
    dataset=f.read()
dataset=dataset.lower().split('\n')
    
tokenizer=Tokenizer()
tokenizer.fit_on_texts(dataset)
max_id=len(tokenizer.word_index)+1
input_sequences=[]
for i in dataset:
    ingram_text=tokenizer.texts_to_sequences([i])[0]
    total_length=len(ingram_text)
    for j in range(0,total_length):
        sequence=ingram_text[0:j+1]
        input_sequences.append(sequence)
        
max_pad=max([len(x) for x in input_sequences])
final_sequences=pad_sequences(input_sequences,max_pad)

xs=final_sequences[:,:-1]
ys=final_sequences[:,-1:]
y_label=tensorflow.keras.utils.to_categorical(ys,max_id)

model=Sequential([Embedding(max_id,64,input_length=max_pad-1),Bidirectional(LSTM(32)),Dense(max_id,activation='softmax')])
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(xs,y_label,epochs=100,verbose=2)