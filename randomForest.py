__author__ = 'Gabriel'

from loading import loadData, loadTrainAndTestFeaturesData

import numpy as np
import time
import scipy as sc

from sklearn.ensemble import RandomForestClassifier

remove6first = lambda x:x[6:]
power = lambda x:[z**2 for z in x[:6]] + [z**3 for z in x[:6]]

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False,power)

storeTime = []
storeScore = []

for i in range(0,100):
    print(i)
    start = time.time()

    randomF = RandomForestClassifier(n_estimators=10)
    randomF.fit(X_train,y_train)

    meanScorerandomF = randomF.score(X_test,y_test)

    storeTime.append(time.time()-start)
    storeScore.append(meanScorerandomF)
    #print("The mean score is : "+str(meanScorerandomF))
    #print("Elapsed time : "+str(time.time()-start))

print(sc.mean(storeTime))
print(sc.std(storeTime))
print(sc.mean(storeScore))
print(sc.std(storeScore))