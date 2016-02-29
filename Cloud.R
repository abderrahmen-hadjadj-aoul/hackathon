# Chargement des données

cloud.data <- read.csv("Small_data_cloud.csv", header = F, colClasses = "numeric")
colnames(cloud.data) <- c("R/G1", "R/B1", "R/I1", "G/B1", "G/I1", "B/I1",
                          "R/G2", "R/B2", "R/I2", "G/B2", "G/I2", "B/I2",
                          "R/G4", "R/B4", "R/I4", "G/B4", "G/I4", "B/I4")

cloud.label <- read.csv("Small_label_cloud.csv", header = F, colClasses = "numeric")
cloud.label[,1] <- as.factor(cloud.label[,1])
colnames(cloud.label) <- c("Cloud")

# Chargement des librairies

library(caret)
library(randomForest)
library(nnet)
source("CloudFunctions.R")

# Partage des sets

index.train <- createDataPartition(1:nrow(cloud.data), p = 0.80, times = 1)[[1]]

train.features <- cloud.data[index.train, ]
train.target <- cloud.label[index.train, ]

test.features <- cloud.data[-index.train, ]
test.target <- cloud.label[-index.train, ]

### RANDOM FORESTS
print("==== RANDOM FORESTS ====")
tRF <- Sys.time()
# Paramètres

nbTrees <- 40

# Training

RF <- randomForest(train.features, train.target, ntree = nbTrees)

# Tests

T_train <- predictTable(model = RF, truth = train.target, train.features)
print(T_train)
print(paste("Précision d'entrainement =", calcPrecision(T_train)))

T_test <- predictTable(model = RF, truth = test.target, test.features)
print(T_test)
print(paste("Précision de test =", calcPrecision(T_test)))

print(Sys.time() - tRF)
print("______________________")

### ANN
print("==== NEURAL NETWORKS ====")
tANN <- Sys.time()
# Paramètres

nbNeurons <- 12
lambda <- 0.3

# Training

train.target.mat <- labelAsMatrix(train.target)
colnames(train.target.mat) <- c("Cloud0", "Cloud1")
test.target.mat <- labelAsMatrix(test.target)
colnames(test.target.mat) <- c("Cloud0", "Cloud1")

scaling.tools <- scaleTable(train.features)

ANN <- nnet(scaling.tools$normData, train.target.mat, size = nbNeurons, decay = lambda, 
            softmax = TRUE, maxit = 50000, trace = FALSE)

train.prediction <- predict(ANN, scaling.tools$normData)
train.prediction <- matrixAsLabels(train.prediction)
train.table <- table(data.frame(prediction = train.prediction, truth = train.target))
print(train.table)
print(paste("Training precision =", calcPrecision(train.table)))

test.prediction <- predict(ANN, scaleTable(test.features, vMin = scaling.tools$vMin, vMax = scaling.tools$vMax)$normData)
test.prediction <- matrixAsLabels(test.prediction)
test.table <- table(data.frame(prediction = test.prediction, truth = test.target))
print(test.table)
print(paste("Testing precision =", calcPrecision(test.table)))

print(Sys.time() - tANN)
print("______________________")