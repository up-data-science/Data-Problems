
#first we call all packages
library(tidyr) 
library(data.table)
library(tensorflow)
library(keras)
library(dplyr)
library(stringr) 
library(forcats) 
library(caret)
library(h2o)
library(plotly)
library(data.table) 
library(dplyr) 
library(readr) 
library(rlang) 
library(tibble) 
library(gridExtra)

#KAYRROS Challenge 
setwd("/Users/miguelangelalbaacosta/Desktop/KAYRROS")

#Preprocessing and exploring the datasets
TARGET = fread("/Users/miguelangelalbaacosta/Desktop/KAYRROS/25.csv")
train = fread("/Users/miguelangelalbaacosta/Desktop/KAYRROS/25/Train.csv")
test = fread("/Users/miguelangelalbaacosta/Desktop/KAYRROS/25-2/Test.csv")


#add the target variable
dtrain = merge(train,TARGET,by= "ID")
str(dtrain) #now that it seems that is no categorical variables just the country and month

######################################
#Overview: file structure and content
######################################

#lets see how it is the train data base just for look if its required oversampling with the target variable

summary(dtrain); summary(test)

#missing values
sum(is.na(dtrain)) #4497
sum(is.na(test)) #0
sapply(train, function(x) sum(is.na(x)))

#homogeneous treatment to all data set for exploring analyses 
comb = bind_rows(dtrain %>% mutate(dset = "dtrain"), test %>% mutate(dset = "test", target = NA))
comb = comb %>% mutate(dset = factor(dset))
#set mounth and country as a factor (categorical) variables
comb$month = as.factor(comb$month) 
comb$country = as.factor(comb$country)

#look if its an unbalanced dataset
dtrain %>%
  ggplot(aes(Target, fill = Target)) +
  geom_bar() +
  theme(legend.position = "none")

dtrain %>%
  group_by(Target) %>%
  summarise(percentage = n()/nrow(dtrain)*100)

#there will be no problem caused by unbalanced data (target) so we dont require ROSE or SPOTE oversampling

# exploring the number of NA's
nano = comb %>%
  is.na() %>%
  rowSums() %>%
  as.integer()

comb = comb %>%
  mutate(nano = nano)

plot.nas = comb %>%
  ggplot(aes(nano, fill = as.factor(nano))) +
  geom_bar() +
  scale_y_log10() +
  theme(legend.position = "none") +
  labs(x = "Number of NA's")
plot.nas


#there are mostly 1 number of NA's per row and 2 next to this number
#we can see the max number of NA's per ID is 25 and its about 500 ids
#at least every row has 1 NA

#some exploring to the dataset before we make the models, with the distribution per country of some of the oil product

#some correlations with features

#diffClosing stocks(kmt)
p11 <- dtrain %>%
  ggplot(aes(`1_diffClosing stocks(kmt)`, `12_diffClosing stocks(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#diffExports(kmt)
p21 <- dtrain %>%
  ggplot(aes(`1_diffExports(kmt)`, `12_diffExports(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#diffImports(kmt)
p31 <- dtrain %>%
  ggplot(aes(`1_diffImports(kmt)`, `12_diffImports(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#diffRefinery intake(kmt) 
p41 <- dtrain %>%
  ggplot(aes(`1_diffRefinery intake(kmt)`, `12_diffRefinery intake(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#`1_diffWTI`
p51 <- dtrain %>%
  ggplot(aes(`1_diffWTI`, `12_diffWTI`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#`1_diffSumClosing stocks(kmt)`
p61 <- dtrain %>%
  ggplot(aes(`1_diffSumClosing stocks(kmt)`, `12_diffSumClosing stocks(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#`1_diffSumExports(kmt)`
p71 <- dtrain %>%
  ggplot(aes(`1_diffSumExports(kmt)`, `12_diffSumExports(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#$`1_diffSumImports(kmt)`
p81 <- dtrain %>%
  ggplot(aes(`1_diffSumImports(kmt)`, `12_diffSumImports(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#`1_diffSumProduction(kmt)`
p91 <- dtrain %>%
  ggplot(aes(`1_diffSumProduction(kmt)`, `12_diffSumProduction(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

#`1_diffSumRefinery intake(kmt)`
p101 <- dtrain %>%
  ggplot(aes(`1_diffSumRefinery intake(kmt)`, `12_diffSumRefinery intake(kmt)`)) +
  geom_point() +
  geom_smooth(method = 'gam', color = "red")

grid.arrange(p11,p21,p31,p41,p51,p61,p71,p81,p91,p101,ncol=5)

#we can visualize a little correlation between the data in periods in every feature.

library(corrplot)
quartz()
dtrain %>% select(-ID) %>%
  mutate(Target = as.integer(Target)) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot(tl.cex = 0.25, tl.col = 'black', type="lower",  diag=FALSE)

#we can see that the strongest correlation between some variables in the diferent periods, 
#de difference in refinery intake with diff sum production, diff sum ex mports and diff in sum imports
#and strong inverse relationships like exportations and importations.

#so once we know there is no binary or categorical features we preceed to make the models
#in this time we used H2o extention of R 

h2o.init(max_mem_size = "10G")
train = as.h2o(dtrain[,-1])
test = as.h2o(test)

splits = h2o.splitFrame(train,0.8,seed = 1234)

train <- h2o.assign(splits[[1]], "train.hex") 
valid <- h2o.assign(splits[[2]], "valid.hex")

 # Identify predictors and response

y <- "Target"
## the response variable is an integer, we will turn it into a categorical/factor for binary classification
train[[y]] <- as.factor(train[[y]]) 
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
valid[,y] <- as.factor(valid[,y])

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

#gbm 
simple_gbm = h2o.gbm(x = x, y = y, 
                     training_frame = train,
                     validation_frame = valid,
                     keep_cross_validation_predictions = TRUE,
                     nfolds = 5 , seed = 1)
# This gives you an idea of the variance between the folds
simple_gbm@model$cross_validation_metrics_summary
h2o.auc(h2o.performance(simple_gbm,newdata = valid))
#A simple model makes an AUC of 0.775071 
# second gbm model
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  ntrees = 100,
                  distribution = "bernoulli",
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "AUTO",
                  keep_cross_validation_predictions = TRUE,
                  ignore_const_cols = TRUE,
                  seed = 1)


#just another gradient boosting machine with more parameters, it has an AUC of 0.786259 with validation data

my_gbm_97 = h2o.gbm(
          x = x, 
          y = y, 
          training_frame = train, 
          validation_frame = valid,
          
          ## more trees is better if the learning rate is small enough 
          ntrees = 10000,                                                            
          
          ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)
          learn_rate=0.01,                                                         
          
          ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
          stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
          sample_rate = 0.8,                                                    
          col_sample_rate = 0.8,                                                   
          seed = 1234,                
          ## score every 10 trees to make early stopping reproducible 
          score_tree_interval = 10)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
                          validation_frame = valid,
                          nfolds = nfolds,
                          fold_assignment = "AUTO",
                          keep_cross_validation_predictions = TRUE,
                          ignore_const_cols = TRUE,
                          seed = 1)

# Train & Cross-validate a DNNº
my_dl <- h2o.deeplearning(x = x,
                          y = y,
                          training_frame = train,
                          l1 = 0.001,
                          l2 = 0.001,
                          hidden = c(200, 200, 200),
                          nfolds = nfolds,
                          fold_assignment = "AUTO",
                          keep_cross_validation_predictions = TRUE,
                          ignore_const_cols = TRUE,
                          seed = 1)

######################
#####XGBoost models #
######################

# Train & Cross-validate a (shallow) XGB-GBM
my_xgb1 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = train,
                       distribution = "bernoulli",
                       ntrees = 50,
                       max_depth = 3,
                       min_rows = 2,
                       learn_rate = 0.2,
                       nfolds = nfolds,
                       fold_assignment = "AUTO",
                       keep_cross_validation_predictions = TRUE,
                       ignore_const_cols = TRUE,
                       seed = 1)


# Train & Cross-validate another (deeper) XGB-GBM
my_xgb2 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = train,
                       distribution = "bernoulli",
                       ntrees = 50,
                       max_depth = 8,
                       min_rows = 1,
                       learn_rate = 0.1,
                       sample_rate = 0.7,
                       col_sample_rate = 0.9,
                       nfolds = nfolds,
                       fold_assignment = "AUTO",
                       keep_cross_validation_predictions = TRUE,
                       ignore_const_cols = TRUE,
                       seed = 1)
# more complex XGB model 
xgb <- h2o.xgboost(x = x
                   ,y = y
                   ,training_frame = train
                   ,validation_frame = valid
                   ,stopping_rounds = 3
                   ,stopping_metric = "AUC"
                   ,distribution = "bernoulli"
                   ,score_tree_interval = 1
                   ,learn_rate=0.1
                   ,nfolds = nfolds
                   ,ntrees=50
                   ,subsample = 0.75
                   ,colsample_bytree = 0.75
                   ,tree_method = "hist"
                   ,grow_policy = "lossguide"
                   ,booster = "gbtree"
                   ,gamma = 0.0
                   ,keep_cross_validation_predictions = TRUE
                   ,ignore_const_cols = TRUE
                   ,seed = 1)


# Train a stacked ensemble using the H2O and XGBoost models from above
base_models <- list(my_gbm@model_id, my_rf@model_id, my_dl@model_id,  
                    my_xgb1@model_id, my_xgb2@model_id, xgb@model_id, simple_gbm@model_id)

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = base_models)

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = valid)

# Compare to base learner performance on the test set
get_auc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = valid))
baselearner_aucs <- sapply(base_models, get_auc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)

print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))
h2o.auc(h2o.performance(ensemble, newdata = valid))


