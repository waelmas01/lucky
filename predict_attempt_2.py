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
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/predict_encode.csv', 'r') as csvfile: #changed to input just for one run online... use output_file.csv
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([int(val) for val in row])
        num_of_rows=num_of_rows+1

#print(num_of_rows)
array_1=[]
for a1 in range(num_of_rows):
    for b1 in range(20):
        array_1.append(data[a1][b1])
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

"""
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

"""

def generate_data(array_1):
	# generate sequence
    sequence = array_1

    encoded = one_hot_encode(sequence)

    print("encoded")
    print(encoded)
	# create lag inputs
    df = DataFrame(encoded)
    
    df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
    
    #at X1,Y1 there is a table with 0s & 1s size(1x80)
    #so X1 has 5 tables (0..4) which are of 0s &1s
    # remove non-viable rows
    values = df.values
    values = values[5:,:]
    print("values all")
    print(values)
    print("df")
    print(df)
    
    print("X shape before encode")
    print(values.shape)
    

    X = values.reshape(3515, 5, 80)
    print("X")
    print(X)
    print("X shape")
    print(X.shape)
    
    
    #sequence_2 = array_1
    #sequence_2 = np.delete(sequence_2, 0)
    #sequence_2 = np.delete(sequence_2, 0)
    #sequence_2 = np.delete(sequence_2, 0)
    #sequence_2 = np.delete(sequence_2, 0)
    #sequence_2 = np.delete(sequence_2, 0)

    #encoded_2 = one_hot_encode(sequence_2)

    print()
    
    print("encoded 22222222")
    #print(encoded_2)
    #print(encoded_2.shape)

    #y=encoded_2

    print("y shape before encode")
    #print(y.shape)

    print("y")
    #print(y)
    print("y shape")
    #print(y.shape)

    return X#, y


#X,y=generate_data(array_1)
print("X")
#print(X)
print("y")
#print(y)

#X,encoded=generate_data(array_1)


filepath = "simple_encoded.h5"
model = load_model(filepath)


#model.save(filepath)
# evaluate model on new data
#array_2=[40,73,72,34,69]
X = generate_data(array_1)
"""
#X_simple=[73,72,34,69,79]

#array_simple=np.array(X_simple)
array_simple=np.array(array_1)
array_simple=one_hot_encode(array_simple)
array_simple=array_simple.reshape(len(array_simple), 5, 80)
"""

yhat = model.predict_proba(X, batch_size=3515) #1

#yhat = model.predict(X)
# we run x+1 for each element in the list to counter the -1 shift we made during encoding


#array_2=np.delete(array_1, 0)
yhat_decoded=one_hot_decode(yhat)
yhat_decoded_correct=[ x+1 for x in yhat_decoded]
#array_2=array_2.astype(int)
#yhat=int(yhat)
print("yhat")
print(yhat_decoded_correct)

"""
array_2=np.delete(array_1, 0)


len_y=yhat_decoded_correct.__len__()
print("len")
print(len_y)
y1=yhat_decoded_correct[len_y-1]

array_2 = np.append(array_2, y1, axis=None)

print("array_2 shape")
print(array_2.shape)

X, y = generate_data(array_2)
yhat2= model.predict(X, batch_size=3515)
"""




#yhat = model.predict(X, batch_size=3515)

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

