#collect script timing information for fun
from datetime import datetime as dt
beginning=dt.now()


#****DATA PROCESSING


#**function sliceData(low, high)
#    - This function will divide the sampleData and sampleClass arrays based on 
#      the sampleMeta array and a range of samples defined in the parameters
#    - For each user, 8 sessions were conducted where 50 samples were collected
#      per session, resulting a total of 400 samples per user low and high 
#      refer to the range of samples from 1 to 400
#    - This function takes two indices, high and low and 
#      returns samples in [low, high] for each user
def sliceData(low, high):
    dataArr=sampleData[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)))]
    classArr=sampleClass[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)))]
    return dataArr, classArr

#define constants
PROJECT_ROOT = 'D:\\Data\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
#PROJECT_ROOT = 'K:\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
DATASET_PATH = [PROJECT_ROOT + '\\datasets\\DSL-StrongPasswordData.csv']

#**setup output
f=open(PROJECT_ROOT+"\\output.txt","w+")
f.write("ExperimentStart\r\n")


import pandas as pd
#read data into dataframe
df = pd.read_csv(DATASET_PATH[0])

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
#for all samples 
for s in range(10,391,10):
    
    # use initialSamplesSize as the number of samples to train the model with at first
    initialSamplesSize=s
    testSamplesSize=10
    #select samples from the dataset for initial model training
    modelData, modelClass = sliceData(1,initialSamplesSize)
    #****INITIATE MODEL
    from sklearn.neighbors import KNeighborsClassifier
    # some values of k (k neighbors)
    for k in range(1,10,2):
        #initialize classifier
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric='manhattan')
        
        #iterations=0
        #accuracy = 0
        #beginning=dt.now()
        
        #****KNN MANHATTEN TESTING
    
        #****FOR EACH SET OF testSamplesSize SAMPLES
        # - Incrementally take testSamplesSize new samples and test
        #   for proper classification.
        # - If the sample is classified properly, add it to the model
        #   Discard the sample otherwise
        # - Get metrics
        # - Update Model
            
        for i in range(initialSamplesSize+1,400,testSamplesSize):
            
            #select test data
            testData, testClass = sliceData(i,i + testSamplesSize - 1)
            
            #train model
            knn.fit(modelData, modelClass)
            
            #run with test data
            pred = knn.predict(testData)
            scor = knn.score(testData,testClass)
            
            #store size of model before adding successful test samples
          #  sizeBefore=len(modelClass)
            #for each sample tested
            
            for j in range(int(len(pred)/testSamplesSize)):
                userScore = 0
                for p in range(testSamplesSize):
                    #if classified correctly
                    if(pred[j*testSamplesSize + p]==testClass[j*testSamplesSize + p] ):
                        #add test sample to model
                        modelData=np.vstack((modelData,testData[j*testSamplesSize+p]))
                        modelClass=np.append(modelClass, testClass[j*testSamplesSize+p])
                        #print(j*testSamplesSize+p)
                        userScore += 1
               # print("k", k, "isize", s, "user", testClass[j*testSamplesSize + p], "i", i, "score", userScore)
                f.write("k %d isize %d user %s i %d score %d\n" %( k, s, testClass[j*testSamplesSize + p], i, userScore))
                    
            #record size of model after adding successful test samples
         #   sizeAfter=len(modelClass)
                
            #calculate accuracy score
            from sklearn.metrics import accuracy_score
            #aScore = accuracy_score(testClass, pred)
            #print('tested with samples ', i, ' through ', i+testSamplesSize-1, ' added', (sizeAfter-sizeBefore) ,'score', aScore)
           # iterations += 1
            #accuracy += aScore
        #print("k ", k, "Average accuracy", (accuracy/iterations), 'initial size', s, 'time', dt.now()-beginning)
f.write("ExperimentEnd")
f.close()