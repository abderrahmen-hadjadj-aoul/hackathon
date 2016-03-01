from loading import loadTrainAndTestFeaturesData, loadFile, cleanData

__author__ = 'Gabriel'

def writeDownSVMFile(valuesData,valuesLabels, dataFilePath):
    f = open(dataFilePath,"w")

    featuresNumber = len(valuesData[0])

    formatString = "{0}"

    for i in range(1,featuresNumber+1):
        formatString += " "+str(i)+":{data["+str(i-1)+"]}"
    formatString +="\n"

    print(formatString)

    for i,l in enumerate(valuesData):
        toWrite = formatString.format(valuesLabels[i],data=l)
        f.write(toWrite)

    f.close()


x,y = loadFile("./data/Small_data_cloud.csv","./data/Small_label_cloud.csv",delimiter=",")
x = cleanData(x)

writeDownSVMFile(x,y, "./Small_data_cloud.svm")
