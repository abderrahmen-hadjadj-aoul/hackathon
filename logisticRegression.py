'''
For LogReg :
Against LogReg :
'''

__author__ = 'Gabriel'

import numpy as np
import time

from sklearn.linear_model import LogisticRegression


start = time.time()

ada = LogisticRegression()
ada.fit(X,y)

meanScoreLogReg = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreLogReg))
print("Elapsed time : "+str(time.time()-start))