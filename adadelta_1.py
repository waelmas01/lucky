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


# univariate convlstm example
from numpy import array
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
import pandas as pd
import math
import csv
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
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


#read CSV and turn into 1 big array
########################################################################################
num_of_rows=0
data = []
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August_no_head.csv', 'r') as csvfile:
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
n_steps_initial = 20
# split into samples
X, y = split_sequence(array_1, n_steps_initial)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 4
n_steps = 5
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
print("x shape")
print(X.shape)


# define model
model = Sequential()
#model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
#model.add(Flatten())
#model.add(Dense(1))
model.add(ConvLSTM2D(filters=512, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='Adadelta', loss='mean_squared_error', metrics=["accuracy"])
#model.compile(optimizer='adam', loss='mse')

# define the checkpoint
filepath = "model_adadelta.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit model #initially verbose was 0 so if problems occur put it back
model.fit(X, y, epochs=2000, verbose=1, callbacks=callbacks_list)


# demonstrate prediction
x_input = array([70.,48.,54.,14.,21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.])
#initial_x_input=x_input
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
#x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
print("input shape")
print(x_input.shape)



yhat = model.predict(x_input, verbose=1)
print("prediction")
print(yhat)

x_input = array([48.,54.,14.,21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.,yhat])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat1 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat1)

x_input = array([54.,14.,21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.,yhat,yhat1])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat2 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat2)

x_input = array([14.,21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.,yhat,yhat1,yhat2])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat3 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat3)

x_input = array([21.,58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.,yhat,yhat1,yhat2,yhat3])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat4 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat4)

x_input = array([58.,64.,8.,46.,25.,21.,48.,78.,12.,76.,70.,3.,2.,71.,51.,yhat,yhat1,yhat2,yhat3,yhat4])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat5 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat5)
print("expected")
print(54,30,15,59,29)


###########################
print()
print("Predictions for 2")
x_input = array([68.,74.,16.,53.,79.,63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.])
#initial_x_input=x_input
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
#x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
print("input shape")
print(x_input.shape)

yhat = model.predict(x_input, verbose=1)
print("prediction")
print(yhat)

x_input = array([74.,16.,53.,79.,63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.,yhat])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat1 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat1)

x_input = array([16.,53.,79.,63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.,yhat,yhat1])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat2 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat2)

x_input = array([53.,79.,63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.,yhat,yhat1,yhat2])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat3 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat3)

x_input = array([79.,63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.,yhat,yhat1,yhat2,yhat3])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat4 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat4)

x_input = array([63.,17.,21.,40.,19.,58.,37.,50.,9.,73.,64.,28.,77.,33.,32.,yhat,yhat1,yhat2,yhat3,yhat4])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat5 = model.predict(x_input, verbose=1)
print("prediction")
print(yhat5)









print("expected")
print(16,64,17,3,58,6,1,2,8,24,39,80,31,28,18,73)