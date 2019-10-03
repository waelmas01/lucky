# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:00:31 2019

USE this for testing that everything with the data size and shapes are correct

first read the CSV file and make one huge array out of it

this will be used to create a model that takes 1 input and gives 1 output
using time sequences as a definer

so technically it will take x(t-n),x(t-n+1),x(t-n+2)... and give an output of y(t+1),y(t+2)...

actually it will try to find a pattern in the generator by assuming all numbers are in the order they
have been generated

@author: W
"""

#added tensorflow.keras instead of keras to try fix an issue with gcloud tensor2.0
# univariate convlstm example
from numpy import array
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D
import pandas as pd
import math
import csv
from numpy.testing import assert_allclose
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
#from sympy import symbols, Eq, solve

# split a univariate sequence into samples
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
NOTES TO MYSELF


first dimension of training data reshaping is size of total data-n_steps

n_steps(initial) * n_features * n_seq * n_steps(aftersplit) SHALL be equal to size of initial data



testing input should be of size=n_features * n_seq * n_steps(after splitting)
?
""" 
# define input sequence
#raw_seq = [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140]


#using a small file at first to test everything 
#dataset = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August_expected.csv')
#dataset = dataset.astype('float32')

#print(dataset.head())
print()

#D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/July_no_head.csv
#read CSV and turn into 1 big array
########################################################################################
num_of_rows=0
data = []
with open('july_1st.csv', 'r') as csvfile: #changed to input just for one run online... use output_file.csv
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([float(val) for val in row])
        num_of_rows=num_of_rows+1

#print(num_of_rows)
array_1=[]
for x in range(num_of_rows):
    for y in range(20):
        array_1.append(data[x][y])
#print(array_1)
#########################################################################################







####### n_steps_initial = n_seq*n_steps(after)
# choose a number of time steps
#was 20 and 25
n_steps_initial = 40
# split into samples
X, y = split_sequence(array_1, n_steps_initial)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
#n_seq was tested 20 and 25
n_seq = 8
n_steps = 5
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
print("x shape")
print(X.shape)


# define model
model = Sequential()
#model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
#model.add(Flatten())
#model.add(Dense(1))
#512 filters
#activation relu and Dense 1
model.add(ConvLSTM2D(filters=256, kernel_size=(1,2), activation='softmax', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=["accuracy"])
#model.compile(optimizer='adam', loss='mse')

# define the checkpoint
filepath = "model_adam.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit model #initially verbose was 0 so if problems occur put it back
#epochs 1000
model.fit(X, y, epochs=500, verbose=1, callbacks=callbacks_list)

#filepath = "model2.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]
"""
# load the model
new_model = load_model("model.h5")
assert_allclose(model.predict(X),
                new_model.predict(X),
                1e-5)

# fit the model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(X, y, epochs=500, verbose=1, callbacks=callbacks_list)
"""

# demonstrate prediction
#x_input = array([77.,38.,3.,78.,72.,70.,48.,54.,14.,21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.])
#x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
#x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
#print("input shape")
#print(x_input.shape)






#yhat = model.predict(x_input, verbose=0)
#print("prediction")
#print(yhat)
print("use predict.py to load model model_adam.h5 and make predictions")
