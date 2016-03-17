from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from loading import loadFile, cleanData
import time
import numpy as np

__author__ = 'Gabriel'

from sklearn import decomposition

x,y = loadFile("./data/Big_data_cloud_SUPAERO.csv","./data/Big_label_cloud_SUPAERO.csv",delimiter=",")
x = cleanData(x)

x_small = np.genfromtxt("./data/Small_data_cloud.csv", delimiter=",")
y_small = np.genfromtxt("./data/Small_label_cloud.csv", delimiter=",")

pca = decomposition.PCA(n_components=7, copy=True, whiten=False)
pca.fit(x_small)

print(pca.explained_variance_ratio_)

x_transformed = pca.transform(x)
print(str(x_transformed.shape))

X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.25, random_state=42)


start = time.time()
randomF = RandomForestClassifier(n_estimators=20,max_depth=10)
randomF.fit(X_train,y_train)

meanScorerandomF = randomF.score(X_test,y_test)

print("The mean score is : "+str(meanScorerandomF))
print("Elapsed time : "+str(time.time()-start))
