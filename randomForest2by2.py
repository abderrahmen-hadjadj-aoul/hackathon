__author__ = 'Gabriel'

from loading import loadData, loadTrainAndTestFeaturesData, forceProportion
from sklearn.metrics import confusion_matrix


import numpy as np
import time
import scipy as sc

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


remove6first = lambda x:x[6:]
power = lambda x:[z**2 for z in x[6:]] + [z**3 for z in x[6:]]



    X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(False,False,remove6first)


    score = []


    start = time.time()

    randomF = RandomForestClassifier(n_estimators=10,max_depth=10)
    randomF.fit(X_train,y_train)

    meanScorerandomF = randomF.score(X_test,y_test)
    score.append(meanScorerandomF)

    print("The mean score is : "+str(meanScorerandomF))
    print("Elapsed time : "+str(time.time()-start))


