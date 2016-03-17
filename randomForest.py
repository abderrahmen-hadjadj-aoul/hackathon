__author__ = 'Gabriel'

from loading import loadTrainAndTestFeaturesData

import time

from sklearn.ensemble import RandomForestClassifier

remove15 = lambda x:[x[9],x[11],x[12]]
power = lambda x:[z**2 for z in x[6:]] + [z**3 for z in x[6:]]

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(False,False,remove15)

start = time.time()

randomF = RandomForestClassifier(n_estimators=20,max_depth=10)
randomF.fit(X_train,y_train)

meanScorerandomF = randomF.score(X_test,y_test)

print("The mean score is : "+str(meanScorerandomF))
print("Elapsed time : "+str(time.time()-start))