__author__ = 'Gabriel'

from loading import loadData, loadTrainAndTestFeaturesData

import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False)

start = time.time()

randomF = RandomForestClassifier(n_estimators=10)
randomF.fit(X_train,y_train)

meanScorerandomF = randomF.score(X_test,y_test)

print("The mean score is : "+str(meanScorerandomF))
print("Elapsed time : "+str(time.time()-start))