'''
For Bagging :
Against Bagging :
'''

__author__ = 'Gabriel'

import numpy as np
import time

from sklearn.ensemble import BaggingClassifier


start = time.time()

ada = BaggingClassifier()
ada.fit(X,y)

meanScoreBagging = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreBagging))
print("Elapsed time : "+str(time.time()-start))