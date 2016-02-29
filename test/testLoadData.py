import sys
sys.path.append('../')

import loading

(X_train, X_test, y_train, y_test) = loading.loadData('small')

len_X_train = len(X_train)
len_X_test = len(X_test)
len_y_train = len(y_train)
len_y_test = len(y_test)

print('\nSMALL')
print('X_train:' + str(len_X_train))
print('X_test:' + str(len_X_test))
print('y_train:' + str(len_y_train))
print('y_test:' + str(len_y_test))


(X_train, X_test, y_train, y_test) = loading.loadData('big')

len_X_train = len(X_train)
len_X_test = len(X_test)
len_y_train = len(y_train)
len_y_test = len(y_test)

print('\nBIG')
print('X_train:' + str(len_X_train))
print('X_test:' + str(len_X_test))
print('y_train:' + str(len_y_train))
print('y_test:' + str(len_y_test))
