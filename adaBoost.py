'''
For AdaBoost :
Ada boost can give a probability for a class
The mean score is : 0.859878787879
Elapsed time : 6.097348928451538
Against AdaBoost :
'''

__author__ = 'Gabriel'


from loading import loadData, loadTrainAndTestFeaturesData

import numpy as np
import time

from sklearn.ensemble import AdaBoostClassifier

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False)

start = time.time()

ada = AdaBoostClassifier()
ada.fit(X_train,y_train)

meanScoreAda = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreAda))
print("Elapsed time : "+str(time.time()-start))