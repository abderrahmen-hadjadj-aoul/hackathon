import numpy
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import os.path

def loadData(dataType='small') :

    dirPath = ''
    path = dirPath + 'data/Small_data_cloud.csv'
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath

    if dataType == 'small':
        fileNameData = dirPath + 'data/Small_data_cloud.csv'
        fileNameLabel = dirPath + 'data/Small_label_cloud.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")
    elif dataType == 'big':
        fileNameData = dirPath + 'data/Big_data_cloud_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/Big_label_cloud_SUPAERO.csv'
        x = numpy.loadtxt(fileNameData,delimiter=",")
        y = numpy.loadtxt(fileNameLabel,delimiter=",")




    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)


'''
Give the raw data + additionnal features
X,X_test has 18 + #additionnalFeatures columns
Y,Y_test has 1 columns

Take a list of features function
A feature function is a function from R^18 -> R^n

Split input in 75% training set and 25% testing set

return X,Y,X_test,Y_test
'''


def loadTrainAndTestFeaturesData(keepRawFeature=True, scaled=False, *listFeatFunction):
    (X,X_test, Y, Y_test) = loadData()

    if scaled:
        X = preprocessing.scale(X)
        X_test = preprocessing.scale(X_test)

    N_F = getNumberArguments(*listFeatFunction)

    print("Total Number of new features : " + str(N_F))
    # Extend X and X_test

    X_feat = extendArray(X, keepRawFeature, N_F)
    X_test_feat = extendArray(X_test, keepRawFeature, N_F)

    if keepRawFeature:
        column = 18
    else:
        column = 0

    X_feat = appliedFeatures(X, X_feat, column, *listFeatFunction)
    X_test_feat = appliedFeatures(X_test, X_test_feat, column, *listFeatFunction)

    return X_feat, X_test_feat,Y, Y_test


'''
Fill up the features
Assume the matrice X as the good size
'''


def appliedFeatures(RawFeatures, whereToAdd, columnStart, *list_features):
    for lineNb, l in enumerate(RawFeatures):
        column = columnStart
        for f in list_features:
            addFeat = f(l[0:18])
            nbFeat = len(addFeat)
            whereToAdd[lineNb, column:(column + nbFeat)] = addFeat
            column += nbFeat
    return whereToAdd


'''
Compute the number of features created by the list of features function
'''


def getNumberArguments(*args):
    z = numpy.zeros(18)
    totalLength = 0

    for f in args:
        additionnalFeaturesNumber = len(f(z))
        totalLength += additionnalFeaturesNumber
        print(str(f.__name__) + " add " + str(additionnalFeaturesNumber) + " new features.")

    return totalLength


'''
Extend an array of numberOfColumn, keep the original value or not
'''


def extendArray(array, keepValues, numberOfColumn):
    (U, V) = array.shape

    if keepValues:
        array_ext = numpy.zeros((U, V + numberOfColumn))
        if not (numberOfColumn == 0):
            array_ext[:, :-numberOfColumn] = array
        else:
            array_ext = array
    else:
        array_ext = numpy.zeros((U, numberOfColumn))

    return array_ext;