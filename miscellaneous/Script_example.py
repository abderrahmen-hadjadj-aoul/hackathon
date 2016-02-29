#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: mlegoff
# @Date:   2016-02-05 09:08:33
# @Last Modified by:   mlegoff
# @Last Modified time: 2016-02-05 09:51:16

import numpy as np

import pyspark as sp
from pyspark.sql.types import *
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.ml import Pipeline
import time
import yaml

""" Launching Spark """

sc = sp.SparkContext()
sql = sp.SQLContext(sc)

""" Setting data path """

path_import_label = "./Small_label_cloud.csv"
path_import_data = "./Small_data_cloud.csv"

""" Reading data and dataframing it """

RDD_data    = sc.textFile(path_import_data)
RDD_data    = RDD_data.map(lambda l: Vectors.dense([np.float(num) for num in l.split(',')]))

RDD_label   = sc.textFile(path_import_label).map(lambda x: float(x))
RDD         = RDD_label.zipWithIndex().map(lambda x : (x[1], x[0])).join(RDD_data.zipWithIndex().map(lambda x:(x[1], x[0])))
RDD         = RDD.map(lambda x: x[1])

DF = sql.createDataFrame(RDD, ['Label', 'unormedFeatures'])

print DF.count()

""" Machine learning pipeline definition """

ss = StandardScaler(withMean=True, withStd=True, inputCol="unormedFeatures", outputCol="features")

s = StringIndexer(inputCol="Label", outputCol="label")

rf = RandomForestClassifier(numTrees=10,
                    featureSubsetStrategy="log2",
                    impurity="entropy",
                    maxDepth=5,
                    maxBins=32,
                    )

pipeline = Pipeline(stages=[ss, s, rf])

""" Computing internal parameters """

to = time.time()

pip = pipeline.fit(DF)

t_fit = time.time() - to

""" Computing statistics on the predictions """

predictionAndLabel = pip.transform(DF).select("prediction","label").map(lambda row: (row.prediction,row.label))

metrics = MulticlassMetrics(predictionAndLabel)

to = time.time()

cm = metrics.confusionMatrix().toArray().tolist()

t_cm = time.time() - to

""" Saving some numbers """

result = dict({
	"data_samples":path_import_data,
	"data_labels":path_import_label,
	"fittingTime":t_fit,
	"statsTime":t_cm,
	"confusionMatrix":cm
	})

yaml.dump(result, open("results.yaml","wb"))