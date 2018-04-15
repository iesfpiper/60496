#collect script timing information for fun
from datetime import datetime as dt
import random
from sklearn.neighbors import KNeighborsClassifier

#****DATA PROCESSING
#**function sliceData(low, high)
#    - This function will divide the sampleData and sampleClass arrays based on 
#      the sampleMeta array and a range of samples defined in the parameters
#    - For each subject, 8 sessions were conducted where 50 samples were collected
#      per session, resulting a total of 400 samples per subject low and high 
#      refer to the range of samples from 1 to 400
#    - This function takes two indices, high and low and 
#      returns samples in [low, high] for each subject
def sliceData(low, high, exclude=None, numRandom=-1):
    dataArr=sampleData[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)) & (sampleClass[:] != exclude))]
    classArr=sampleClass[(((((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) >=low) & (((sampleMeta[:,0]-1)*50 + sampleMeta[:,1]) <=high)) & (sampleClass[:] != exclude))]
    if numRandom>0:
        randDataArr=np.empty((0,len(dataArr[0]))).astype('float_')
        randClassArr=np.empty(0).astype('unicode_')
        for x in random.sample(range(0,len(classArr)),numRandom):
            randDataArr=np.vstack((randDataArr,dataArr[x]))
            randClassArr=np.append(randClassArr,classArr[x])
        return randDataArr, randClassArr
        
    return dataArr, classArr

#define constants
#PROJECT_ROOT = 'D:\\Data\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
PROJECT_ROOT = 'K:\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
DATASET_PATH = [PROJECT_ROOT + '\\datasets\\DSL-StrongPasswordData.csv']

samplesPerSubject = 400

#**setup output
f=open(PROJECT_ROOT+"\\output\\output8s.txt","w+")
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
numSubjects = len(sampleClassUnique)

#****EXPERIMENT
#Loop through a range of initialSamplesSize
for initialSamplesSize in range(10,301,10):
    samplesPerTrial=10
	
    #select samples from the dataset for initial model training
    modelData, modelClass = sliceData(1,initialSamplesSize)
	
    #****INITIATE MODEL
    # some values of k (k neighbors)
    for k in range(1,10,2):
		#For each user, track the number of samples in the model 
        numSamples=np.zeros(numSubjects).astype('int')
        
        #initialize classifier
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric='manhattan')
        
        
        #****KNN MANHATTEN TESTING
    
        #****FOR EACH SET OF samplesPerTrial SAMPLES
        # - Incrementally take samplesPerTrial new samples and test
        #   for proper classification.
        # - If the sample is classified properly, add it to the model
        #   Discard the sample otherwise
        # - Get metrics
        # - Update Model by training on newly added samples
        
		# all samples in intervals of samplesPerTrial until 310 samples per user, or less
		# - i is the lower index from where we take our next 10 samples
            #train model
        knn.fit(modelData, modelClass)
        
        for i in range(initialSamplesSize+1,310,samplesPerTrial):
            
            #select test data
            testData, testClass = sliceData(i,i + samplesPerTrial - 1)
            
            
            #run with test data
            pred = knn.predict(testData)
                        
            #for each subject, j
            for j in range(int(len(pred)/samplesPerTrial)):
                trueAcceptanceCount = 0
                
				#for each sample in this trial for user j, add properly classified samples to model
                for p in range(samplesPerTrial):
                    #if classified correctly
                    if(pred[j*samplesPerTrial + p]==testClass[j*samplesPerTrial + p] ):
                        #add test sample to model
                       # modelData=np.vstack((modelData,testData[j*samplesPerTrial+p]))
                       # modelClass=np.append(modelClass, testClass[j*samplesPerTrial+p])
                        #print(j*samplesPerTrial+p)
                        trueAcceptanceCount += 1
                
                #calculate FAR
                imposterSamples = 400
                far=0
                imposterData, imposterClass = sliceData(i + samplesPerTrial, samplesPerSubject, sampleClassUnique[j], imposterSamples)
                imposterPred=knn.predict(imposterData)
                far += len(imposterPred[(imposterPred[:]==sampleClassUnique[j])])
                
                    
               
                f.write("k %d isize %d subject %s i %d samplesInModel %d score %d farAccept %d imposterSamples %d\n" %( k, initialSamplesSize, testClass[j*samplesPerTrial + p], i, initialSamplesSize + numSamples[j], trueAcceptanceCount, far, imposterSamples)) 
                print("k %d isize %d subject %s i %d samplesInModel %d score %d farAccept %d imposterSamples %d" %( k, initialSamplesSize, testClass[j*samplesPerTrial + p], i, initialSamplesSize + numSamples[j], trueAcceptanceCount, far, imposterSamples)) 
              
                numSamples[j] += trueAcceptanceCount          
            #record size of model after adding successful test samples
         #   sizeAfter=len(modelClass)
                
            #calculate accuracy score
            from sklearn.metrics import accuracy_score
            #aScore = accuracy_score(testClass, pred)
            #print('tested with samples ', i, ' through ', i+samplesPerTrial-1, ' added', (sizeAfter-sizeBefore) ,'score', aScore)
           # iterations += 1
            #accuracy += aScore
        #print("k ", k, 'initial size', initialSamplesSize)
f.write("ExperimentEnd")
f.close()