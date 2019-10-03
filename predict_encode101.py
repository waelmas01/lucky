from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
import csv
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
array_2=[]
for x in range(num_of_rows):
    for y in range(20):
        array_2.append(data[x][y])
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

def generate_data(array_n):
	# generate sequence
	sequence = array_n
	# one hot encode
	encoded = one_hot_encode(sequence)
	# create lag inputs
	df = DataFrame(encoded)
    
	df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
    
	# remove non-viable rows
	values = df.values
	values = values[5:,:]
    
	# convert to 3d for input
	X = values.reshape(len(values), 5, 80)
	# drop last value from y
	#y = encoded[4:-1,:]
	return X







filepath = "simple_encoded.h5"

model = load_model(filepath)

model.save(filepath)
# evaluate model on new data
array_2=[1,1,1,1,1,1,22,9,42,65,66,5,46,19,6,31,58,7,52,11]
X= generate_data(array_2)
print(X)
yhat = model.predict(X, batch_size=5)

# we run x+1 for each element in the list to counter the -1 shift we made during encoding
#y_decoded=one_hot_decode(y)
#y_decoded_correct=[ x+1 for x in y_decoded]

yhat_decoded=one_hot_decode(yhat)
yhat_decoded_correct=[ x+1 for x in yhat_decoded]

#print('Expected:  %s' % y_decoded_correct)
print('Predicted: %s' % yhat_decoded_correct)