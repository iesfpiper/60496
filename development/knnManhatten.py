

#****DATA PROCESSING
import pandas as pd
projectRoot = 'D:\\Data\\Documents\\School\\Current\\60496\\Python\\Repository\\60496'
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
modelData = sampleData[(sampleMeta[:,0]==1) & (sampleMeta[:,1] <= 10)]
modelClass = sampleClass[(sampleMeta[:,0]==1) & (sampleMeta[:,1] <= 10)]

#remaining entries
testData = sampleData[(sampleMeta[:,0]==1) & (sampleMeta[:,1] > 10) | (sampleMeta[:,0]>1)]
testClass = sampleClass[(sampleMeta[:,0]==1) & (sampleMeta[:,1] > 10) | (sampleMeta[:,0]>1)]

#****INITIATE MODEL
from sklearn.neighbors import KNeighborsClassifier
#initialize model
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric='manhattan')

#train model
knn.fit(modelData, modelClass)

pred = knn.predict(testData)

from sklearn.metrics import accuracy_score
aScore = accuracy_score(testClass, pred)
#****KNN MANHATTEN TESTING

    #****FOR EACH SET OF 10 SAMPLES
    # Incrementally take 10 new samples and test for proper classification
    # If the sample is classified properly, add it to the model
    # Discard the sample otherwise
    # Get metrics
    # Update Model
    