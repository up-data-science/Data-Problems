library(DMwR)
library(smotefamily)
library(ROSE)
library(stringr)
##### ROSE
library(caret)

#make a partition of 80% with out H2o
trainIndex <- createDataPartition(dtrain$Target, p = 0.8, list = FALSE, times = 1)

train <- dtrain[trainIndex,]
valid <- dtrain[-trainIndex,]

#quit the ID 
train = train[,-1]
valid = valid[,-1]
#preprocesing the names of the variables because of some issues with the ROSE oversampling algorithm parse code
s = colnames(train)
ss = str_replace_all(s,"_diff","dd")
ss = str_replace_all(ss, " ",".")
ss = str_replace_all(ss, "´(kmt)´","ktm")
ss = str_replace_all(ss, "\\s*\\([^\\)]+\\)","")
ss = str_sub(ss, 3, str_length(ss))

ss = ss[-c(1:2)] #with out mounth and country

sec = c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10),rep(6,10),rep(7,10),rep(8,10)
        ,rep(9,10),rep(10,10),rep(11,10),rep(12,10))

sec = as.character(sec)
sec = c(sec,"1n")

sss = paste(ss, sec, sep="")
sss = c(s[c(1:2)],sss)
sss[123] = "Target"

colnames(train) = sss
colnames(valid) = sss 
train <- ROSE(Target~., data = train, seed = 1)$data
table(train$Target)
prop.table(table(train$Target)) # as we can see there is about the same number of 1's and 0's

##### h2o 
h2o.init(max_mem_size = "10G")
train = as.h2o(train)
test = as.h2o(test)
valid = as.h2o(valid)

# Identify predictors and response
y <- "Target"
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
valid[,y] <- as.factor(valid[,y])

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  distribution = "bernoulli",
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "AUTO",
                  keep_cross_validation_predictions = TRUE,
                  ignore_const_cols = TRUE,
                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
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

xgb <- h2o.xgboost(x = x
                   ,y = y
                   ,training_frame = train
                   ,validation_frame = valid
                   ,stopping_rounds = 3
                   ,stopping_metric = "logloss"
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
                    my_xgb1@model_id, my_xgb2@model_id, xgb@model_id)

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
h2o.auc(h2o.performance(my_xgb1,newdata = valid))

