'''
For Bagging :
The mean score is : 0.856363636364
Elapsed time : 8.814503908157349
Against Bagging :
'''

__author__ = 'Gabriel'

from loading import loadData

import numpy as np
import time

from sklearn.ensemble import BaggingClassifier

(X_train, X_test, y_train, y_test) = loadData('small')


start = time.time()

ada = BaggingClassifier()
ada.fit(X_train,y_train)

meanScoreBagging = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreBagging))
print("Elapsed time : "+str(time.time()-start))