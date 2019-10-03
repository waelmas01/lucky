# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:17:18 2019

@author: W
"""

# univariate convlstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

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

""" 
# define input sequence
raw_seq = [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140]
# choose a number of time steps
#n_steps = 4
n_steps = 5
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
#model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
#model.add(Flatten())
#model.add(Dense(1))
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([110,120,130,140])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
#x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)