# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:22:27 2019

Better predict.py version for auto input and prediction

@author: W
"""

#added tensorflow.keras instead of keras to try fix an issue with gcloud tensor2.0

# commenting imports of keras to be able to run it in Spyder and check data
from numpy import array
#from keras.layers import Flatten
#from keras.layers import ConvLSTM2D
import pandas as pd
import math
import csv
from numpy.testing import assert_allclose
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


#D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/July_no_head_test.csv
num_of_rows=0
data = []
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/predict.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([float(val) for val in row])
        num_of_rows=num_of_rows+1

#file path for trained model to load
filepath = "D:/Brand Personal/Programming/Machine Learning/Lucky games/trained_on_cloud/simple_encoded.h5"

n_steps_initial = 40
#had it set to 20 and 25 before

n_features = 1
n_seq = 8
n_steps = 5

print()
input_row=int(input("Number of row as input:    (counting starts from 1)"))
input_row=input_row-1
print()
num_of_predictions=int(input("Number of desired future predictions: "))
print()

# load the model
new_model = load_model(filepath)

#assert_allclose(new_model.predict(X),
#                new_model.predict(X),
#                1e-5)

yhat=3.0

#########################################################################################
var_in=[]
list_1=data[input_row]


x_input = array(list_1)
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = new_model.predict(x_input, verbose=1)
print("prediction")
yhat=int(yhat)
print(yhat)



for x101 in range (num_of_predictions-1):
    list_1.remove(list_1[0])
    list_1.append(yhat)
    x_input = array(list_1)
    x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
    yhat = new_model.predict(x_input, verbose=1)
    yhat=int(yhat)
    print(yhat)

