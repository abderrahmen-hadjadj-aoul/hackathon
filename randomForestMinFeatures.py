__author__ = 'Gabriel'

from loading import loadTrainAndTestFeaturesData, appliedFeatures

import time

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools as it

remove6first = lambda x:[x[12]]
keep6 = lambda x,y: [x[i] for i in y]

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False)

u,v = X_train.shape
u_test,v_test = X_test.shape

X_feat = np.zeros((u,6))
X_test_feat = np.zeros((u_test,6))

#Need to define each keep 6
variablesCombinationsToTest = it.combinations([x for x in range(0,18)],3)

score = []
combination = []

for c in variablesCombinationsToTest:

    keep6 = lambda x: [x[i] for i in c]

    X_feat = appliedFeatures(X_train, X_feat, 0, keep6)
    X_test_feat = appliedFeatures(X_test, X_test_feat, 0,keep6)

    start = time.time()

    randomF = RandomForestClassifier(n_estimators=20,max_depth=10)
    randomF.fit(X_feat,y_train)

    meanScorerandomF = randomF.score(X_test_feat,y_test)
    score.append(meanScorerandomF)
    combination.append(c)

    print("The mean score is : "+str(meanScorerandomF))
    print("Elapsed time : "+str(time.time()-start))
    print("Combi"+str(c) + " score :" + str(meanScorerandomF))

MaxcombinationIndex = score.index(max(score))

print("Score max :")
print(max(score))
print("Combi")
print(combination[MaxcombinationIndex])