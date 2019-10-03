from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
import csv
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pandas as pd


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
#read CSV and turn into 1 big array
########################################################################################
num_of_rows=0
data = []
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/predict_encode.csv', 'r') as csvfile: #changed to input just for one run online... use output_file.csv
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([int(val) for val in row])
        num_of_rows=num_of_rows+1

#print(num_of_rows)
array_2=[]
for x in range(num_of_rows):
    for y in range(20):
        array_2.append(data[x][y])
#print(array_1)
#########################################################################################
"""

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


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



def generate_data(array_1):
	# generate sequence
    sequence = array_1

    encoded = one_hot_encode(sequence)

    #print("encoded")
    #print(encoded)
	# create lag inputs
    df = DataFrame(encoded)
    
    df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
    
    #at X1,Y1 there is a table with 0s & 1s size(1x80)
    #so X1 has 5 tables (0..4) which are of 0s &1s
    # remove non-viable rows
    values = df.values
    values = values[5:,:]
    #print("values all")
    #print(values)
    #print("df")
    #print(df)
    
    #print("X shape before encode")
    #print(values.shape)
    

    X = values.reshape(len(values), 5, 80)
    #print("X")
    #print(X)
    #print("X shape")
    #print(X.shape)

    sequence_2 = array_1
    sequence_2 = np.delete(sequence_2, 0)
    sequence_2 = np.delete(sequence_2, 0)
    sequence_2 = np.delete(sequence_2, 0)
    sequence_2 = np.delete(sequence_2, 0)
    sequence_2 = np.delete(sequence_2, 0)

    encoded_2 = one_hot_encode(sequence_2)

    #print()
    
    #print("encoded 22222222")
    #print(encoded_2)
    #print(encoded_2.shape)

    y=encoded_2
    y = y.reshape(3515,80)
    
    #print("y shape before encode")
    #print(y.shape)

    #print("y")
    #print(y)
    #print("y shape")
    #print(y.shape)

    return X, y


X,y=generate_data(array_1)
#y = y.reshape(3515,1,80)

print("X")
print(X)
print("shape of X")
print(X.shape)
print("y")
print(y)
print("shape of y")
print(y.shape)

#X,encoded=generate_data(array_1)
x_decoded=one_hot_decode(X)
x_decoded_correct=[ x+1 for x in x_decoded]

#y_decoded=one_hot_decode(y)
#y_decoded_correct=[ x+1 for x in y_decoded]

y_decoded=one_hot_decode(y)
y_decoded_correct=[ x+1 for x in y_decoded]


print('Input:  %s' % x_decoded_correct)
print(x_decoded_correct.__len__())

print()
#print('Expected:  %s' % y_decoded_correct)
print()
print('Output: %s' % y_decoded_correct)
print(y_decoded_correct.__len__())

print()


filepath = "simple_encoded.h5"
#model = load_model(filepath)

# define model
model = Sequential()
model.add(LSTM(80, batch_input_shape=(3515, 5, 80), stateful=True)) # 5,5,80 or #1,5,80
#model.add(LSTM(80, input_shape=(5, 80)))

#model.add(LSTM(50, stateful=True)) #no need to specify input shape again
#model.add(LSTM(50, stateful=True))
model.add(Dense(80, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model

for i in range(500): #2000
    X, y = generate_data(array_1)
    model.fit(X, y, epochs=1, batch_size=3515, verbose=2, shuffle=True)#False #But True might be better, batch 5
    model.save(filepath)
    model.reset_states()

#model.fit(X, y, epochs=50, verbose=2)

