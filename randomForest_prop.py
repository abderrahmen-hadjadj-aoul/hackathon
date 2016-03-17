__author__ = 'Gabriel'

from loading import loadData, loadTrainAndTestFeaturesData, forceProportion, forceProportionDuplication
from sklearn.metrics import confusion_matrix


import numpy as np
import time
import scipy as sc

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


remove6first = lambda x:x[6:]
power = lambda x:[z**2 for z in x[6:]] + [z**3 for z in x[6:]]

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(False,False,remove6first)

confusionArray = []
score = []
proportion = [float(x)/10. for x in range(1,8)]

for prop in proportion:
    X_train,y_train = forceProportionDuplication(X_train,y_train,proportionClass1=prop)

    start = time.time()

    randomF = RandomForestClassifier(n_estimators=20)
    randomF.fit(X_train,y_train)

    meanScorerandomF = randomF.score(X_test,y_test)
    score.append(meanScorerandomF)
    y_pred = randomF.predict(X_test)
    confusionArray.append(confusion_matrix(y_test, y_pred,labels=[1.0,0.0]) / float(len(y_test)) )

    print("The mean score is : "+str(meanScorerandomF))
    print("Elapsed time : "+str(time.time()-start))

tp = [x[0,0] for x in confusionArray]
tn = [x[1,1] for x in confusionArray]
fp = [x[1,0] for x in confusionArray]
fn = [x[0,1] for x in confusionArray]

tpr = [x[0,0]/(x[0,0]+x[0,1]) for x in confusionArray]
tnr = [x[1,1]/(x[1,1]+x[1,0]) for x in confusionArray]
fpr = [x[1,0]/(x[1,1]+x[1,0]) for x in confusionArray]
fnr = [x[0,1]/(x[0,0]+x[0,1]) for x in confusionArray]

print(score)
print(tpr)
print(tnr)

plt.plot(proportion,tpr,'g',proportion,tnr,'r',proportion,fnr,'b',proportion,fpr,'black',proportion,score,'cyan')
plt.legend(["TP rate","TN rate","FP rate","FN rate","Score"])
plt.show()