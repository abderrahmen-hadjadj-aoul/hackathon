import numpy
from sklearn.cross_validation import train_test_split
import os.path

def loadData(dataType) :

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
    else:
        fileNameData = dirPath + 'data/Big_data_cloud_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/Big_label_cloud_SUPAERO.csv'

    x = numpy.genfromtxt(fileNameData, delimiter=",")
    y = numpy.genfromtxt(fileNameLabel, delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)
