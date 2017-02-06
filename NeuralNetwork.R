##########@doc implementing forward pass##################
#Random Alpha weight for X value to hidden layer 
AlphaWeight <- function(Xvalue,m,p){ 
  Alpha = matrix(,nrow = m , ncol = p,byrow = TRUE)
  for (i in 1:m)  {
    Alpha[i,] <- runif(p,-0.7,0.7)
  }
  return(Alpha)
}

#Random Beta weight from hidden layer to output value
BetaWeight <- function(m){
  beta <- runif(m,-0.7,0.7)
  return(beta)
}

#Sigmoid function to calculate in hidden layer
Sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

#calculating Z hidden layer
HiddenNetwork <- function(Alpha,m,Xvalue,AlphaBias){
  hiddenValue = NULL
  for(j in 1:m ){
    AlphaSum=sum(Alpha[j,] * Xvalue) #value of X from data set Auto[1,]
    hiddenValue[j]=  Sigmoid(AlphaBias[j]+AlphaSum) # consider bias
  }
  return(hiddenValue)
}

#calculating Y Output
CalculatedY <- function(Beta,hiddenValue,BetaBias,m){
  BetaSum <- 0
  for(i in 1:m){
    BetaSum <- BetaSum + Beta[i]*hiddenValue[i]
  }
  #print(BetaSum)
  calcY =  BetaBias+ BetaSum 
  #print(calcY)
  return (calcY)
}

#################################implementing backpropagation#############

#calculating Sigmoid Prime for 
SigmoidPrime <- function(x){
  Fx <- Sigmoid(x)
  sigmoidPrime <- Fx*(1- Fx) #(exp(x))/((exp(x)+1)^2)
  return(sigmoidPrime)
}

#calculating delta error between real output and calculated output 
ErrorD <- function(realY,Y){
  return(2* (realY - Y ))
}

#calculating error S that means implementing backpropragation equation

ErrorS <- function(Beta,Alpha,Xvalue,errorD,m,AlphaBias){
  errorS = NULL
  for(j in 1:m ){
    AlphaSum=sum(Alpha[j,] * Xvalue)
    x <- AlphaBias[j]+AlphaSum
    errorS[j]<- SigmoidPrime(x) * (Beta*errorD)
  }
  return(errorS)
}

#############@Updating Weight##############
UpdatedBeta <- function(Beta,hiddenValue,errorD,number){ #learning rate 0.001
  return(Beta -(0.001/number)* (errorD*hiddenValue))
}

UpdatedAlpha <- function(Alpha,m,errorS,Xvalue,p,number){ #learning rate 0.001 and considering number of observations
  updatedAlpha = matrix(,nrow = m , ncol = p,byrow = TRUE)
  for(i in 1:m){
    updatedAlpha[i,] <- (-0.001/number) * errorS *t( Xvalue) + Alpha[i,]
    
  }
  return(updatedAlpha)
}
############## Main function ##################
NeuralNet <- function(DataSourceX,DataSourceY,m){
  set.seed(0790)
  DataSourceX <- scale(DataSourceX)
 # DataSourceY <- scale(DataSourceY)
  DataSourceX.col <- ncol(DataSourceX)
  Datasource <- data.frame(DataSourceX,DataSourceY)
  train_Data <- sample(1:nrow(Datasource),nrow(Datasource)/2)
  test_Data <- Datasource[-train_Data,]
  test_DataX <- test_Data[,1:DataSourceX.col]
  test_DataY <- test_Data[,(DataSourceX.col+1)]
  DataSourceX <- Datasource[,1:DataSourceX.col]
  DataSourceY <- Datasource[,(DataSourceX.col+1)]
  Xvalue <- DataSourceX[1,] #observation attributes
  realY <- DataSourceY[1] #our real label
  Parameter <- ncol(DataSourceX) # number of input attributes
  Alpha <- AlphaWeight(Xvalue,m,Parameter) #Calculate random Alpha for the first time
  AlphaBias <- runif(m,-0.7,0.7) #considering bias as random number for first time
  BetaBias <- runif(1,-0.7,0.7) #considering beta bias as random number for first time
  Beta <- BetaWeight(m)   #calculating beta as random for first time
  hiddenValue <- HiddenNetwork(Alpha,m,Xvalue,AlphaBias) #calculating hidden neuron value
  Y <- CalculatedY(Beta,hiddenValue,BetaBias,m) #calculating label Y
  calculatedYmatrix <- matrix(,nrow = nrow(DataSourceX),ncol = 1)
  calculatedYmatrix[1]<- Y
  errorD <- ErrorD(realY,Y)  #calculating error between predicted value and real value
  errorS <- ErrorS(Beta,Alpha,Xvalue,errorD,m,AlphaBias) #calculating sim
  updatedBeta <- UpdatedBeta(Beta,hiddenValue,errorD,1) #changing beta weight
  updatedAlpha <- UpdatedAlpha(Alpha,m,errorS,Xvalue,Parameter,1) #changing alpha's weight
  UpdatedBetaBias <- 0
    
    for(n in 1:100 ){ #doing back propragation for n munber of times
      Xvalue <- DataSourceX[n,]
      realY <- DataSourceY[n]
      UpdatedBetaBias <- BetaBias -(0.001/n)* sum(errorD*hiddenValue)
      UpdatedAlphaBias <- AlphaBias -(0.001/n)*(errorS*Xvalue)
      hiddenNetwork <- HiddenNetwork(updatedAlpha,m,Xvalue,AlphaBias)
      Y <- CalculatedY(updatedBeta,hiddenNetwork,UpdatedBetaBias,m)
      calculatedYmatrix[n]<- Y
      errorD <- ErrorD(realY,Y)
      errorS <- ErrorS(updatedBeta,updatedAlpha,Xvalue,errorD,m,AlphaBias)
      updatedBeta <- UpdatedBeta(updatedBeta,hiddenNetwork,errorD,n)
      updatedAlpha <- UpdatedAlpha(updatedAlpha,m,errorS,Xvalue,Parameter,n)
      #################Implementing stopping condition if ten continous mse are in 1% change range
    }
  
  # print(updatedBeta)
  #printing final value of all parameters
  train.MSE <- 0
  for(n in 1:100){
    train.MSE <- train.MSE + sum(DataSourceY[n] -calculatedYmatrix[n] )^2
  }
  train.MSE <- train.MSE/100
  print(c("train.MSE:",train.MSE))
  
  ################## feed forward
  for(k in nrow(test_DataX)){
    Xvalue.test <-test_DataX[k,] 
    hiddenValue_test <- HiddenNetwork(updatedAlpha,m,Xvalue.test,UpdatedAlphaBias)
    PredictY <- matrix(,nrow= nrow(test_Data),ncol =1)
    for(i in 1:nrow(test_Data)){
      Y <- CalculatedY(updatedBeta,hiddenNetwork,UpdatedBetaBias,m)
      PredictY[i, ] <- Y
    }
  }
  x <- NULL
  for(i in 1:100){
    x[i] <-(test_Data[i,1]-PredictY[i,])^2
  }
  return(x)

}

library(ISLR)
#Creating boxplot for different M
x1<- NeuralNet(Auto[,c(4:5,7:8)],Auto[,1],1)
x2 <- NeuralNet(Auto[,c(4:5,7:8)],Auto[,1],2)
x3 <- NeuralNet(Auto[,c(4:5,7:8)],Auto[,1],3)
x4 <- NeuralNet(Auto[,c(4:5,7:8)],Auto[,1],4)
#warnings()
dataM<-data.frame(M1=x1,M2=x2,M3=x3,M4=x4)
boxplot(dataM,ylab="Test MSE",xlab="Number of Hidden Node",col=c("red","blue","yellow","sienna"))

# Creating boxplot for different number of hidden layers 


