## 
library(e1071)
library(caret)
trainIndex <- createDataPartition(dtrain$Target, p = 0.8, list = FALSE, times = 1)

train <- dtrain[trainIndex,]
valid <- dtrain[-trainIndex,]

#quit the ID 
train = train[,-1]
valid = valid[,-1]


library(nlme)
registerDoMC(cores = 2)
# Set seeds for reproducibility 
## In this case B = (5 repeats of 5-Fold CV) +1 = 51; M = 1 (only one parameter combination being used)
set.seed(1)
seeds1 <- vector(mode = "list", length = 26)
for(i in 1:25) seeds1[[i]] <- sample.int(1000, 25) ## 5*5 parameters from tuneLength=5
## For the last model:
seeds1[[26]] <- sample.int(1000, 1)

dtrain$Target = as.factor(dtrain$Target)
train$Target = as.factor(train$Target)
valid$Target = as.factor(valid$Target)
#levels
levels(dtrain$Target) <- c('down', 'up')
levels(train$Target) <- c('down', 'up')
levels(valid$Target) <- c('down', 'up')

cntr = trainControl(method = "repeatedcv",
                    number = 1,
                    repeats = 1,
                    classProbs = TRUE,
                    summaryFunction = twoClassSummary,
                    savePredictions = TRUE,
                    seeds = seeds1) # optional

train1 = train(Target~., train,
               method = "svmRadial",
               # train() use its default method of calculating an analytically derived estimate for sigma
               tuneLength = 1,# 5 arbitrary values for C and sigma = 25 models
               trControl = cntr,
               preProc = c("center", "scale"),
               metric = "ROC",
               verbose = FALSE,na.action=na.exclude)

 max(train1$results[,"ROC"])
library(pROC)
 # ROC using plotROC (ggplot2 extension)
 gc_prob_ex <- extractProb(list(train1), valid[,-123])[,1]
 gc_ggROC <- ggplot(gc_prob_ex, aes(d=obs, m=Good)) + geom_roc() 
 gc_ggROC_styled <- gc_ggROC +  annotate("text", x = .75, y = .25, 
                                         label = paste("AUC =", round(calc_auc(gc_ggROC)$AUC, 2)))
 gc_ggROC_styled

 
 
 
 
 
 
 
######
 
# load the library
library(mlbench)
library(caret)

 # calculate correlation matrix
 correlationMatrix <- cor(dtrain[,4:124],use = "complete.obs") 
 # summarize the correlation matrix
 print(correlationMatrix)
 # find attributes that are highly corrected (ideally >0.75)
 highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
 # print indexes of highly correlated attributes
 print(highlyCorrelated)
 
 #### Rank features with importance
 
 # prepare training scheme
 control <- trainControl(method="repeatedcv",classProbs = TRUE, number=10, repeats=3)
 # train the model
 model <- train(Target~., data=train, method="svmRadial", preProcess="scale", tuneLength = 1, metric = "ROC", trControl=control,na.action=na.exclude)
 # estimate variable importance
 importance <- varImp(model, scale=FALSE)
 # summarize importance
 print(importance)
 # plot importance
 plot(importance)
 
 #recursive feature elimination
 
 # define the control using a random forest selection function
 control <- rfeControl(functions=rfFuncs, method="cv", number=10)
 # run the RFE algorithm
 results <- rfe(train[,1:122],train$Target, sizes=c(1:123), rfeControl=control)
 # summarize the results
 print(results)
 # list the chosen features
 predictors(results)
 # plot the results
 plot(results, type=c("g", "o"))
 
 
 
 