'''
For LogReg :
The mean score is : 0.821515151515
Elapsed time : 0.14900898933410645
Against LogReg :
'''

__author__ = 'Gabriel'

from loading import loadData

import numpy as np
import time

from sklearn.linear_model import LogisticRegression

(X_train, X_test, y_train, y_test) = loadData('small')


start = time.time()

ada = LogisticRegression()
ada.fit(X_train,y_train)

meanScoreLogReg = ada.score(X_test,y_test)

print("The mean score is : "+str(meanScoreLogReg))
print("Elapsed time : "+str(time.time()-start))