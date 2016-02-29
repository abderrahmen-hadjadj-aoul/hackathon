'''
For AdaBoost : Ada boost can give a probability for a class
Against AdaBoost :
'''

__author__ = 'Gabriel'

import numpy as np
import time

from sklearn.ensemble import AdaBoostClassifier


start = time.time()

ada = AdaBoostClassifier()
ada.fit(X,y)

meanScoreAda = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreAda))
print("Elapsed time : "+str(time.time()-start))