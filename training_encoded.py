# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:25:58 2019

@author: W
"""

from random import randint
from numpy import array
from numpy import argmax
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
#D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/July_no_head.csv
#read CSV and turn into 1 big array
########################################################################################
num_of_rows=0
data = []
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/july_1st.csv', 'r') as csvfile: #changed to input just for one run online... use output_file.csv
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([int(val) for val in row])
        num_of_rows=num_of_rows+1

#print(num_of_rows)
array_1=[]
for x in range(num_of_rows):
    for y in range(20):
        array_1.append(data[x][y])
#print(array_1)
#########################################################################################



"""
# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]
"""



#So now in order to have a shape of 80 instead of 81 (easier for usage with reshaping)
#We shift everything -1 so table is 0-79 but the real thing is everything+1
# one hot encode sequence
def one_hot_encode(sequence, n_unique=80):
    encoding = list()
    
    for value in sequence:
        vector=[0]*80
        #vector = [0 for _ in range(n_unique)]
        _ = 0
        for _ in range(n_unique):
            vector[_]=0
        vector[value-1] = 1
        encoding.append(vector)
    return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    
	return [argmax(vector) for vector in encoded_seq]



# generate data for the lstm
def generate_data(array_1,timesteps=1):
	# generate sequence
	sequence = array_1
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to 3d for input
	X = encoded.reshape(encoded.shape[0], timesteps, encoded.shape[1])
	return X, encoded


X,encoded=generate_data(array_1)

print(X.shape)
#print(encoded)
decoded=one_hot_decode(encoded)
# we run x+1 for each element in the list to counter the -1 shift we made during encoding
decoded_correct=[ x+1 for x in decoded]
#print(decoded)


filepath = "simple_encoded.h5"

# define model
model = Sequential()
model.add(LSTM(15, input_shape=(1, 80)))
model.add(Dense(80, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model
for i in range(500):
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    X, y = generate_data(array_1)
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
    model.save(filepath)
# evaluate model on new data
    
model.save(filepath)
X, y = generate_data()
yhat = model.predict(X)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))





