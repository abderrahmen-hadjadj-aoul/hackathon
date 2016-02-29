# Fonctions métriques

predictTable <- function(model, truth, ...)
{
      prediction <- predict(model, ...)
      T_test <- table(data.frame(prediction, truth))
      return(T_test)
}

calcErreur <- function(T_test)
      return(1 - sum(diag(T_test))/sum(T_test))

calcPrecision <- function(T_test)
      return( sum(diag(T_test))/sum(T_test) )

labelAsMatrix <- function(labels)
{
      nbExamples <- length(labels)
      nbLevels <- length(levels(labels))
      
      labels <- as.numeric(labels)
      
      M <- matrix(0, nrow = nbExamples, ncol = nbLevels)
      i <- 1
      for(i in 1:nbExamples)
            M[i, labels[i]] <- 1
      
      return(as.data.frame(M))
}

matrixAsLabels <- function(matrix)
{
      nbExamples <- nrow(matrix)
      nbLevels <- ncol(matrix)
      
      labels <- numeric(nbExamples)
      for(i in 1:nbExamples)
            labels[i] <- which(matrix[i,] == max(matrix[i,])) - 1
      
      return(as.factor(labels))
}

# Normalization functions

scaleTable <- function(data, vMin = NULL, vMax = NULL)
{
      nbCols <- ncol(data)
      flagNotNull <- TRUE
      
      if(is.null(vMin))
      {
            print("null")
            vMin <- numeric(nbCols)
            vMax <- numeric(nbCols)
            flagNotNull <- FALSE
      }
      
      for(j in 1:nbCols)
      {
            if(flagNotNull)
            {
                  data[,j] <- (data[,j] - vMin[j])/(vMax[j] - vMin[j])
            }
            else
            {
                  vMin[j] <- min(data[,j])
                  vMax[j] <- max(data[,j])
                  
                  data[,j] <- (data[,j] - vMin[j])/(vMax[j] - vMin[j])
            }
      }
      
      return(list("normData" = data, "vMin" = vMin, "vMax" = vMax))
}