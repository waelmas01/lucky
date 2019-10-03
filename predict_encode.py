from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
import csv
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
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

    #print("y shape before encode")
    #print(y.shape)

    #print("y")
    #print(y)
    #print("y shape")
    #print(y.shape)

    return X, y


X,y=generate_data(array_1)
print("X")
print(X)
print("y")
print(y)

#X,encoded=generate_data(array_1)


filepath = "simple_encoded.h5"
#model = load_model(filepath)

# define model
model = Sequential()
model.add(LSTM(80, batch_input_shape=(3515, 5, 80), stateful=True)) # 5,5,80

#model.add(LSTM(50, stateful=True)) #no need to specify input shape again
#model.add(LSTM(50, stateful=True))
model.add(Dense(80, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model
for i in range(500): #2000
    X, y = generate_data(array_1)
    model.fit(X, y, epochs=1, batch_size=3515, verbose=2, shuffle=False)#False #But True might be better, batch 5
    model.save(filepath)
    model.reset_states()

"""
#model.save(filepath)
# evaluate model on new data
#array_2=[40,73,72,34,69]
X, y = generate_data(array_2)

X_simple=[73,72,34,69,79]

#array_simple=np.array(X_simple)
array_simple=np.array(array_2)
array_simple=one_hot_encode(array_simple)
array_simple=array_simple.reshape(len(array_simple), 5, 80)

yhat = model.predict(X_simple, batch_size=5)

# we run x+1 for each element in the list to counter the -1 shift we made during encoding

x_decoded=one_hot_decode(X)
x_decoded_correct=[ x+1 for x in x_decoded]

#y_decoded=one_hot_decode(y)
#y_decoded_correct=[ x+1 for x in y_decoded]

yhat_decoded=one_hot_decode(yhat)
yhat_decoded_correct=[ x+1 for x in yhat_decoded]

print('Input:  %s' % x_decoded_correct)
print()
#print('Expected:  %s' % y_decoded_correct)
print()
print('Predicted: %s' % yhat_decoded_correct)
"""


"""
print()
print()
print(bin(40)[1:])
print(bin(73)[1:])
print(bin(72)[1:]) 
print(bin(34)[1:]) 
print(bin(69)[1:]) 
print(bin(79)[1:]) 
print(bin(22)[1:]) 
print(bin(9)[1:])
print(bin(42)[1:])
print(bin(65)[1:])
print(bin(66)[1:]) 
print(bin(5)[1:]) 
print(bin(46)[1:]) 
print(bin(19)[1:]) 
print(bin(6)[1:]) 
print(bin(31)[1:])
print(bin(58)[1:])
print(bin(7)[1:])
print(bin(52)[1:]) 
print(bin(11)[1:]) 


print()
print()
print(hex(40)[1:])
print(hex(73)[1:])
print(hex(72)[1:]) 
print(hex(34)[1:]) 
print(hex(69)[1:]) 
print(hex(79)[1:]) 
print(hex(22)[1:]) 
print(hex(9)[1:])
print(hex(42)[1:])
print(hex(65)[1:])
print(hex(66)[1:]) 
print(hex(5)[1:]) 
print(hex(46)[1:]) 
print(hex(19)[1:]) 
print(hex(6)[1:]) 
print(hex(31)[1:])
print(hex(58)[1:])
print(hex(7)[1:])
print(hex(52)[1:]) 
print(hex(11)[1:]) 

"""
