# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:01:32 2019

Created this file to be able to correct the file DateTime stamp (commented part)
and to produce a Spider graph that nicely shows the frequency of each number from 0-80

Contains other things that are commented.

@author: Wael

"""





from string import ascii_uppercase # to be able to loop throu Alphabet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns #just to test spider chart
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#for spider chart
from math import pi


dataset = pd.read_csv('D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/July_no_head.csv')

#Used to merge data and time into one  (USE IT FOR EVERY FIRST TIME OF A FILE)
#dataset['Time'] = pd.to_datetime(dataset.Date + ' ' + dataset.Time)
#print(dataset.head())
#dataset.to_csv(r'D:/Brand Personal/Programming/Machine Learning/Lucky games/DATA/testing/CSV/kino_full.csv')




#X = dataset.iloc[:, 0:2].values
#Y = dataset.iloc[:, 2:22].values

#Transform into DateTime stamp to be able to use it
#dataset['Time'] = pd.to_datetime(dataset.Time, dayfirst=True)
#print(dataset.head())




#This shows the number of occurences of value 1 in all of A
print(dataset.groupby(['A']).size().to_frame(1).reset_index())

# With all the below we create a list that has how many occurences of each number exist in all the CSV file
counting=[0] * 80
endchar=ord('U') #letter after T
for A in ascii_uppercase:
    if ord(A) < endchar:
        for flag1 in range(80):
            
            # Occurences of #1 in each column
            #dataset.groupby([A]).size().to_frame(1).reset_index()[1][0]
            counting[flag1]=counting[flag1]+dataset.groupby([A]).size().to_frame(1).reset_index()[1][flag1]
    else:
        break



#creating a spider chart to display frequency of each number in all the CSV file

#creating a list of variables from 1 to 80 to use for graph
numlist=[0] * 80
for flag2 in range(80):
    numlist[flag2]=flag2+1
#print(numlist)


#####################################################################################
# FOR CREATING THE SPIDER CHART
    
# Set data
df = pd.DataFrame({
'group': [numlist]

})
    
for flag3 in range(80):
    df[flag3+1]=counting[flag3]



print(df.head())



# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([700,750,800,850,900], ["700","750","800","850","900"], color="grey", size=7)
plt.ylim(700,900)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

####### TO SHOW WHEN RUN FROM TERMINAL
plt.show()


#counting_sorted=counting
#counting_sorted.sort(reverse=True) #to make it ascending




# END OF SPIDER CHART
#####################################################################################



#For looping through alphabet
#for A in ascii_uppercase:
#    print(A)

#print(dataset.Time.dt.hour.head())
#print(dataset.Time.dt.minute.head())
#print(dataset.Time.dt.day.head())
#print(dataset.Time.dt.month.head())
#print(dataset.Time.dt.year.head())



# Split the data between the Training Data and Test Data
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
#                                                    ,random_state = 0)





#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

print()
print()

#a1=dataset.iloc[0]
#a1.astype("float32")
#ar1=array(a1.values)
#print(ar1[0]*1)

"""
print(dataset.A.value_counts().sort_index().plot())
print(dataset.B.value_counts().sort_index().plot())
print(dataset.C.value_counts().sort_index().plot())
print(dataset.D.value_counts().sort_index().plot())
print(dataset.E.value_counts().sort_index().plot())
print(dataset.F.value_counts().sort_index().plot())
print(dataset.G.value_counts().sort_index().plot())
print(dataset.H.value_counts().sort_index().plot())
print(dataset.I.value_counts().sort_index().plot())
print(dataset.J.value_counts().sort_index().plot())
print(dataset.K.value_counts().sort_index().plot())
print(dataset.L.value_counts().sort_index().plot())
print(dataset.M.value_counts().sort_index().plot())
print(dataset.N.value_counts().sort_index().plot())
print(dataset.O.value_counts().sort_index().plot())
print(dataset.P.value_counts().sort_index().plot())
print(dataset.Q.value_counts().sort_index().plot())
print(dataset.R.value_counts().sort_index().plot())
print(dataset.S.value_counts().sort_index().plot())
print(dataset.T.value_counts().sort_index().plot())

"""