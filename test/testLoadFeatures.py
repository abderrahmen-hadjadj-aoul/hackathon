from loading import loadTrainAndTestFeaturesData, cleanData

__author__ = 'Gabriel'

f = lambda x:x[-3:]

X_train, X_test,y_train, y_test = loadTrainAndTestFeaturesData(True,False)

print(X_train[-5:,:])
cleanData(X_train)

print(y_train[-50:])
len_X_train = X_train.shape
len_X_test = X_test.shape
len_y_train = y_train.shape
len_y_test = y_test.shape

print('\nBIG')
print('X_train:' + str(len_X_train))
print('X_test:' + str(len_X_test))
print('y_train:' + str(len_y_train))
print('y_test:' + str(len_y_test))
