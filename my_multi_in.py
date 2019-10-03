# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 01:04:28 2019

@author: W
"""

# multivariate multi-step data preparation
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from string import ascii_uppercase # to be able to loop throu Alphabet

# load the dataset
#dataframe = read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/airline_passengers.csv', usecols=[1], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')

dataset = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August.csv')
#Below is for testing with a very small sample 
#dataset = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August_low.csv')
#dataset = dataset.values
dataset = dataset.astype('float32')

dataset_test = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August_test.csv')
#dataset = dataset.values
dataset_test = dataset_test.astype('float32')

dataset_ex = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/August_expected.csv')
#dataset = dataset.values
dataset_ex = dataset_ex.astype('float32')
print(dataset)

#in_seq1= dataset['A']
#in_seq1= array(in_seq1.values)

#creating lists with names in_seqA, in_seqB, ... in_seqT that has values of the column in dataset 
endchar=ord('U') #letter after T
for A in ascii_uppercase:
    if ord(A) < endchar:
        
        #create arrays from each col
        globals()['in_seq' + A]=dataset[A]   #the globals() is used so we can create a dynamic varible name on the go
        globals()['in_seq' + A]=globals()['in_seq' + A].values
        print(globals()['in_seq' + A])
    else:
        break



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		
        # find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		
        # check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		
        # gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



# define input sequence

in_seq1 = array(in_seqA)
in_seq2 = array(in_seqB)
in_seq3 = array(in_seqC)
in_seq4 = array(in_seqD)
in_seq5 = array(in_seqE)
in_seq6 = array(in_seqF)
in_seq7 = array(in_seqG)
in_seq8 = array(in_seqH)
in_seq9 = array(in_seqI)
in_seq10 = array(in_seqJ)
in_seq11 = array(in_seqK)
in_seq12 = array(in_seqL)
in_seq13 = array(in_seqM)
in_seq14 = array(in_seqN)
in_seq15 = array(in_seqO)
in_seq16 = array(in_seqP)
in_seq17 = array(in_seqQ)
in_seq18 = array(in_seqR)
in_seq19 = array(in_seqS)

out_seq = array(in_seqT)


# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq2), 1))
in_seq4 = in_seq4.reshape((len(in_seq2), 1))
in_seq5 = in_seq5.reshape((len(in_seq2), 1))
in_seq6 = in_seq6.reshape((len(in_seq2), 1))
in_seq7 = in_seq7.reshape((len(in_seq2), 1))
in_seq8 = in_seq8.reshape((len(in_seq2), 1))
in_seq9 = in_seq9.reshape((len(in_seq2), 1))
in_seq10 = in_seq10.reshape((len(in_seq2), 1))
in_seq11 = in_seq11.reshape((len(in_seq2), 1))
in_seq12 = in_seq12.reshape((len(in_seq2), 1))
in_seq13 = in_seq13.reshape((len(in_seq2), 1))
in_seq14 = in_seq14.reshape((len(in_seq2), 1))
in_seq15 = in_seq15.reshape((len(in_seq2), 1))
in_seq16 = in_seq16.reshape((len(in_seq2), 1))
in_seq17 = in_seq17.reshape((len(in_seq2), 1))
in_seq18 = in_seq18.reshape((len(in_seq2), 1))
in_seq19 = in_seq19.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))


# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,
                  in_seq9,in_seq10,in_seq11,in_seq12,in_seq13,in_seq14,in_seq15,in_seq16,
                  in_seq17,in_seq18,in_seq19, out_seq)) #### Define number of arrays as in and out


# choose a number of time steps
n_steps_in, n_steps_out = 20, 5 # last number is how many future guessings we want to make each time
#n_steps_in, n_steps_out = 19, 3

# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)


# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

print("X shape, Y shape")
print(X.shape, y.shape)


# summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])
#for i in range(5):
	#print(X[i], y[i])
#    print(y[i])

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(X, y, epochs=400, verbose=0)





# demonstrate prediction
x_input = array([[dataset_test.iloc[0]],
                 [dataset_test.iloc[1]],
                 [dataset_test.iloc[2]],
                 [dataset_test.iloc[3]],
                 [dataset_test.iloc[4]],
                 [dataset_test.iloc[5]],
                 [dataset_test.iloc[6]],
                 [dataset_test.iloc[7]],
                 [dataset_test.iloc[8]],
                 [dataset_test.iloc[9]],
                 [dataset_test.iloc[10]],
                 [dataset_test.iloc[11]],
                 [dataset_test.iloc[12]],
                 [dataset_test.iloc[13]],
                 [dataset_test.iloc[14]],
                 [dataset_test.iloc[15]],
                 [dataset_test.iloc[16]],
                 [dataset_test.iloc[17]],
                 [dataset_test.iloc[18]],
                 [dataset_test.iloc[19]]])

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)
print(yhat)
print()
print()
print("Expected out: ")
print()
print(array([dataset_ex.iloc[0],dataset_ex.iloc[1],dataset_ex.iloc[2],dataset_ex.iloc[3],dataset_ex.iloc[4]]))

print()

"""
#UNCOMMENT TO SEE DATA FORMATION BEFORE RUNNING THE TRAINING


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		
        
        # find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		
        
        # check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		
        # gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#trying to split my dataset to arrays
#in_seq11 = array(dataset)
#print(in_seq11)

# define input sequence
in_seq1 = array(in_seqA)
in_seq2 = array(in_seqB)
out_seq = array(in_seqT)


# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))


# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))


# choose a number of time steps
n_steps_in, n_steps_out = 3, 2


# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)


# summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])
for i in range(5):
	print(X[i], y[i])  
"""


