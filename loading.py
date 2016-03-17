import numpy
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import os.path

import mmap

def numberLine(file):
    nbLine =0
    for l in file:
        nbLine += 1

    file.seek(0)

    return nbLine

def getShapes(file,delimiter):
    f = open(file)
    nbLine = numberLine(f)

    l = f.readline()
    nbFeatures = l.count(delimiter) + 1
    f.close()

    return (nbLine,nbFeatures)

def loadFile(fileNameData,fileNameLabel,delimiter=","):

    (nbLine,nbFeatures) = getShapes(fileNameData,delimiter)
    x = numpy.zeros((nbLine,nbFeatures), dtype=numpy.float)
    y = numpy.zeros((nbLine),dtype=numpy.float)

    indexRemoved = []
    nbIndexRemoved = 0

    for i,l in enumerate(open(fileNameData,"r")):
        nbIndexRemoved = len(indexRemoved)
        try:
            elements = l.split(delimiter)
            for j,e in enumerate(elements):
                x[i-nbIndexRemoved,j] = float(e)
        except ValueError:
            indexRemoved.append(i)

    x.resize((nbLine-nbIndexRemoved,nbFeatures))
    y.resize((nbLine-nbIndexRemoved))

    ii = 0
    for i,l in enumerate(open(fileNameLabel,"r")):
        if i in indexRemoved:
            pass
        else:
            y[ii] = float("True" in l)
            ii +=1

    return x,y


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
        x,y = loadFile(fileNameData,fileNameLabel,delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)




'''
Remove extremes values (which may have results from dividing by zero)
'''
def cleanData(array, maxValue=1e6, minValue=-1e6, zeroValue=1e-6):
    for i,l in enumerate(array):
        for j,v in enumerate(l):
            if v > maxValue:
                array[i,j] = maxValue
            elif v < minValue:
                array[i,j] = minValue
            elif abs(v) < zeroValue:
                array[i,j] = zeroValue

    return array


'''
Give the raw data + additionnal features
X,X_test has 18 + #additionnalFeatures columns
Y,Y_test has 1 columns

Take a list of features function
A feature function is a function from R^18 -> R^n

Split input in 66% training set and 33% testing set

return X,Y,X_test,Y_test
'''
def loadTrainAndTestFeaturesData(keepRawFeature=True, scaled=False, *listFeatFunction):
    (X,X_test, Y, Y_test) = loadData("big")

    X = cleanData(X)
    X_test = cleanData(X_test)

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

def forceProportion(values,labels,proportionClass1 = 0.5):
    assert proportionClass1 != 0.0

    total = len(labels)
    numberClass1 = sum(labels)
    numberClass0 = total-numberClass1

    newTot = int(numberClass1/proportionClass1)
    class0NumberToKeep = int((1-proportionClass1) * newTot)
    assert class0NumberToKeep + numberClass1 == newTot

    width_array = len(values[0])

    newValues = numpy.zeros((newTot,width_array))
    newLabels = numpy.zeros((newTot))

    i = numberClass1
    j = class0NumberToKeep

    currentLineRead = 0
    currentLineWrite = 0

    while max(i,j) > 0:
        if labels[currentLineRead] == 1.0:
            if(i>0):
                newValues[currentLineWrite] = values[currentLineRead]
                newLabels[currentLineWrite] = labels[currentLineRead]
                i -=1
                currentLineWrite +=1
        else:
            if(j>0):
                newValues[currentLineWrite] = values[currentLineRead]
                newLabels[currentLineWrite] = labels[currentLineRead]
                j -=1
                currentLineWrite +=1

        currentLineRead +=1

    return newValues,newLabels

def forceProportionDuplication(values,labels,proportionClass1 = 0.5):
    assert proportionClass1 != 0.0

    total = len(labels)
    numberClass1 = sum(labels)
    numberClass0 = total-numberClass1

    newTot = int(numberClass0/(1-proportionClass1))

    newNumberClass1 = int(newTot * proportionClass1)

    assert newNumberClass1 + numberClass0 == newTot

    width_array = len(values[0])

    newValues = numpy.zeros((newTot,width_array))
    newLabels = numpy.zeros((newTot))

    i = newNumberClass1
    j = numberClass0

    currentLineRead = 0
    currentLineWrite = 0

    while max(i,j) > 0:
        if currentLineRead < total:

            if labels[currentLineRead] == 1.0:
                if(i>0):
                    newValues[currentLineWrite] = values[currentLineRead]
                    newLabels[currentLineWrite] = labels[currentLineRead]
                    i -=1
                    currentLineWrite +=1
            else:
                if(j>0):
                    newValues[currentLineWrite] = values[currentLineRead]
                    newLabels[currentLineWrite] = labels[currentLineRead]
                    j -=1
                    currentLineWrite +=1

        else:

            if(i>0):
                newValues[currentLineWrite] = newValues[currentLineRead%total]
                newLabels[currentLineWrite] = newLabels[currentLineRead%total]
                i -=1
                currentLineWrite +=1

        currentLineRead +=1

    return newValues,newLabels

