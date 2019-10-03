# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:11:35 2019

taking a csv file (Input_file.csv) and reshaping it to one big column, then saving it as Output_file.csv

@author: W
"""


import csv
import pandas as pd

#setting parameters to DEFAULT in case the script is used without human input
header_flag=1 # set to 0 if header row does NOT exist
num_of_cols=20 # change if number of columns in CSV file is different

# Human input
########################################################################################
"""
 COMMENT this part if the script will be used automatically with no human input
"""
print()
header_exists=input("Header row exists ? y or n     Default:y")
if header_exists == "y":
    header_flag=1
elif header_exists == "n":
    header_flag=0
else:
    print("Assuming header row exists...")
    header_flag=1

print()
num_cols=input("Number of Columns in the file:     Default:20")
if num_cols==True:
    num_of_cols=int(num_cols)
else:
    print("Assuming number of columns is 20...")
    num_of_cols=20

print()
########################################################################################



#read CSV and turn into 1 big array
########################################################################################
num_of_rows=0
data = []
with open('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/Input_file.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([val for val in row])
        num_of_rows=num_of_rows+1


#print(num_of_rows)
array_1=[]
for x in range(num_of_rows-header_flag):
    for y in range(num_of_cols):
        array_1.append(int(data[x+header_flag][y]))
#########################################################################################


# Writing data to a CSV file 1xsize_of_data... one big Column
#########################################################################################
dataset_2=pd.DataFrame(array_1, index=None)
print(dataset_2.head())

dataset_2.to_csv(r'D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/output_file.csv',index = None, header=False)
print()
print("Created one_col_file.csv wich has all the data of input csv turned into one long column.")


