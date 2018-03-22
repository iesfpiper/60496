# -*- coding: utf-8 -*-

import pandas as pd


#read data into a data frame, applying names array created above
df = pd.read_csv('D:\\Data\\Documents\\School\\Current\\60496\\Datasets\\CMU Keystroke Dynamics – Benchmark Data Set\\DSL-StrongPasswordData.csv')
#Call the dataframe's head function, default is to print first 5 rows
df.head()

#Efficient data storage structure (memory, time)
import numpy as np
#from sklearn.model_selection import train_test_split


X=np.array(df)

#X=np.array(df.iloc[:, 3:])

#taking just the class column from the array
y=np.array(df['subject'])

#This will be used to randomly partition the data into train and test data
#from sklearn.cross_validation import train_test_split
#split training and test data
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

X_train = X[X[:,1] <=4]
X_train = X_train[:, 3:]

X_test = X[X[:,1] >4]
X_test = X_test[:, 3:]

y_train = X[X[:,1] <=4]
y_train = y_train[:, 0:1]
y_train = y_train.ravel()

y_test = X[X[:,1] >4]
y_test = y_test[:, 0:1]

#y_train = [x[0] for x in X if x[1] in range(1,5)]
#y_test = [x[0] for x in X if x[1] not in range(1,5)]


#X_train = np.array(df.

from sklearn.neighbors import KNeighborsClassifier
#Create a knn classifier
knn=KNeighborsClassifier(n_neighbors=5, n_jobs=1, metric='mahalanobis', metric_params=dict(V=np.cov(X_train.astype(float))))
#fit the model using training data
knn.fit(X_train, y_train)
#run knn on test data
pred = knn.predict(X_test)

#calculate accuracy of predictions
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, pred))


