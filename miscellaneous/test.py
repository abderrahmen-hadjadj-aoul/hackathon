from loading import loadTrainAndTestFeaturesData

__author__ = 'Gabriel'

import numpy as np

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False)

print(np.amin(X_train))
print(np.amax(X_train))
print(np.amin(X_test))
print(np.amax(X_test))