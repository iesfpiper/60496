import pandas as pd

#Create a list of names which will be used to label columns of imported data
names=['septal_length', 'septal_width', 'petal_length', 'petal_width', 'class']

#read data into a data frame, applying names array created above
df = pd.read_csv('D:\\Data\\Documents\\School\\Current\\60496\\Python\\iris.data.txt', header=None, names=names)
#Call the dataframe's head function, default is to print first 5 rows
df.head()

#Efficient data storage structure (memory, time)
import numpy as np
#from sklearn.model_selection import train_test_split

#some depricated shit, change this later
X=np.array(df.iloc[:, 0:4])

#taking just the class column from the array
y=np.array(df['class'])

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

#Create a list with numbers [1,50)
myList=list(range(1,50))
neighbors=filter(lambda x: x % 2 != 0, myList)

cv_scores = []

from sklearn.model_selection import cross_val_score
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1-x for x in cv_scores]

optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" %optimal_k)

#plt.plot(neighbors, MSE)

