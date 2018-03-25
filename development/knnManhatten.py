
from datetime import datetime as dt
#****DATA PROCESSING
beginning=dt.now()
#Define function
def sliceData(low, high):
    dataArr=sampleData[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)))]
    classArr=sampleClass[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)))]
    return dataArr, classArr

import pandas as pd
#projectRoot = 'D:\\Data\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
projectRoot = 'K:\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'

datasetPath = [projectRoot + '\\datasets\\DSL-StrongPasswordData.csv']

#read data into dataframe
df = pd.read_csv(datasetPath[0])

#slice data and convert to numpy array
# Note that in this section, the following suffexes are defined
#   Data - keystroke timing data
#   Meta - Information about sample collection (i.e. session 1 sample 20)
#   Class - Information identifiying the subject
# The following prefixes are defined
#   sample - general collection of all samples
#   model - Data  or Class for training the model
#   test - Data or Classe for testing
import numpy as np

#keystroke timing data
sampleData = np.array(df.iloc[:,3:]).astype('float_')
#collection metadata
sampleMeta = np.array(df.iloc[:,1:3]).astype('int')
#classification
sampleClass = np.array(df['subject']).astype('unicode_')
sampleClassUnique = np.unique(sampleClass)

#first 10 entries
#modelData = sampleData[(sampleMeta[:,0]==1) & (sampleMeta[:,1] <= 10)]
#modelClass = sampleClass[(sampleMeta[:,0]==1) & (sampleMeta[:,1] <= 10)]
start=100
modelData, modelClass = sliceData(1,start)
for i in range(start+1,400,10):
    
   # modelData, modelClass = sliceData(i,i+9)
    
    #remaining entries
    #testData = sampleData[(sampleMeta[:,0]==1) & (sampleMeta[:,1] > 10) | (sampleMeta[:,0]>1)]
    #testClass = sampleClass[(sampleMeta[:,0]==1) & (sampleMeta[:,1] > 10) | (sampleMeta[:,0]>1)]
    testData, testClass = sliceData(i,i+9)
    #****INITIATE MODEL
    from sklearn.neighbors import KNeighborsClassifier
    #initialize model
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric='euclidean')
    
    #train model
    knn.fit(modelData, modelClass)
    
    pred = knn.predict(testData)
    sizeBefore=len(modelClass)
    for j in range(len(pred)):
        #print("pred",pred[j],' actual',testClass[j])
        if(pred[j]==testClass[j]):
            #print("pred",pred[j],' actual',testClass[j])
            modelData=np.vstack((modelData,testData[j]))
            modelClass=np.append(modelClass, testClass[j])
    sizeAfter=len(modelClass)
        
    
    from sklearn.metrics import accuracy_score
    aScore = accuracy_score(testClass, pred)
    print('iteration', i, 'added(avg)', (sizeAfter-sizeBefore)/51 ,'score', aScore)
    #****KNN MANHATTEN TESTING
    
        #****FOR EACH SET OF 10 SAMPLES
        # Incrementally take 10 new samples and test for proper classification
        # If the sample is classified properly, add it to the model
        # Discard the sample otherwise
        # Get metrics
        # Update Model
        
    #bar=sliceData(low=1,high=1)
print("time took ",dt.now()-beginning)