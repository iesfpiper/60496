import pandas as pd


#read data into a data frame, applying names array created above
df = pd.read_csv('D:\\Data\\Documents\\School\\Current\\60496\\Datasets\\CMU Keystroke Dynamics – Benchmark Data Set\\DSL-StrongPasswordData.csv')
#Call the dataframe's head function, default is to print first 5 rows
df.head()

#Efficient data storage structure (memory, time)
import numpy as np
#from sklearn.model_selection import train_test_split

#some depricated shit, change this later
X=np.array(df.iloc[:, 3:])

#taking just the class column from the array
y=np.array(df['subject'])

#This will be used to randomly partition the data into train and test data
from sklearn.cross_validation import train_test_split
#split training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
#Create a knn classifier
knn=KNeighborsClassifier(n_neighbors=3)
#fit the model using training data
knn.fit(X_train, y_train)
#run knn on test data
pred = knn.predict(X_test)

#calculate accuracy of predictions
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, pred))


