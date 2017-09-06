# set your working directory
path <- "input path"
setwd(path)

# load packages
packages <-c("caret", "plyr", "dplyr", "xgboost", "MLmetrics")  #  load plyr ALWAYS before dplyr, AND not load plyr again.

lapply(packages, FUN = function(X) {
  do.call("library", list(X)) 
})

# memory management
memory.limit(size = 100000)

# load files
data <- readRDS(file = "data/data.Rda")

##
# Prepare data to for xgboost
##

# Fill in column to not import
colExclude <- c("display_address","street_address" , "manager_id",  "building_id_original",  "fullDescription",
                "features", "description", "building_id_new")
data <-  data[, !(colnames(data) %in% colExclude)]

# transform the categorical data to numeric 
data$class <- as.numeric(as.factor(data$interest_level))-1  # convert to levels starting from 0 for xgboost

factorVars <- sapply(data, is.factor)
colnames(data[factorVars])
data[,factorVars] <- as.numeric(unlist(data[,factorVars]))

charVars <- sapply(data, is.character)
colnames(data[charVars])

data$created <- as.numeric(data$created)

# pick variables to model
varExclude <- c("order", "set", "frame","interest_level","lowYN", "midYN", "highYN", "class")
varnames <- setdiff(colnames(data), varExclude)

# separate frame "train" into two sets (train + valid), 50% of data each.  
# Do not use observations in frame = "preProp" to ensure no leak from vtreat encoding
train <- subset(data, data$frame == "train")
index <- createDataPartition(train$class, 
                             p = .5, 
                             list = FALSE, 
                             times = 1)
train <- train[index,]
valid <- train[-index,]
test <- subset(data, data$frame == "test")

# Convert dataset to sparse format
train_sparse <- Matrix(as.matrix(train[varnames]), sparse=TRUE)
valid_sparse <- Matrix(as.matrix(valid[varnames]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(test[varnames]), sparse=TRUE)

# set other labels
listing_id_test <- test$listing_id
labels <- train$class

# Convert data into xgb format
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)
dvalid <- xgb.DMatrix(data= valid_sparse)
dtest <- xgb.DMatrix(data= test_sparse)

##
# Run base model to rank variable importance
##

# Set parameters
param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric= "mlogloss",
              num_class=3,
              eta = .02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .4)

# Run cross-validation
xgbPrepropCv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 10000, 
                 maximize=FALSE,
                 prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                 nfold = 5,
                 stratified = TRUE, 
                 print_every_n = 100,
                 early_stopping_round=300)

# Stopping. Best iteration:
# [1396]	train-mlogloss:0.367981+0.001571	test-mlogloss:0.557841+0.006021

# Run model
watch <- list(dtrain=dtrain)
xgbModel <- xgb.train(data = dtrain,
                  params = param,
                  watchlist=watch,
                  nrounds = xgbPrepropCv$best_ntreelimit,
                  verbose = 1,
                  print_every_n = 100)

##
# Variable Importance / Feature Selection using base model
##
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]],model = xgbModel)
importance$logGain <- log(importance$Gain)
importance$logCover <- log(importance$Cover)
importance$logFrequency <- log(importance$Frequency)
hist(importance$logGain)
hist(importance$logCover)
hist(importance$logFrequency)
cumSumImportance <- importance %>%
  select(Gain:Frequency) %>%
  mutate_each(funs(cumsum))
colnames(cumSumImportance ) <- paste0("cumSum_",colnames(cumSumImportance ))
importance <- cbind(importance, cumSumImportance )
  
# Keep cumulative Gain up to 93%. 
discardFeatures <- filter(importance, cumSum_Gain > 0.92)
x <- c("token_", "treat_", "dist_")
discardTokens <- grep(paste(x, collapse = "|"), discardFeatures$Feature, ignore.case = TRUE)
discardTokensName <- discardFeatures[discardTokens, "Feature"]
notDiscarded <- discardFeatures[-discardTokens, "Feature"]
saveRDS(discardTokensName, file = "data/discardTokensName")

# Trim dataset 
dataTrim <- data[ , -which(names(data) %in% discardTokensName)]
saveRDS(dataTrim, file = ("data/dataTrim.Rda"))

###
# Run model on Trimmed Dataset to confirm performance - performance is similar with half the number of features
###
train <- subset(dataTrim, dataTrim$frame == "train")
index <- createDataPartition(train$class, 
                             p = .5, 
                             list = FALSE, 
                             times = 1)
train <- train[index,]
valid <- train[-index,]
test <- subset(dataTrim, dataTrim$frame == "test")

# pick variables to model
varExclude <- c("order", "set", "frame","interest_level","lowYN", "midYN", "highYN", "class")
varnames <- setdiff(colnames(dataTrim), varExclude)

# Convert dataset to sparse format
train_sparse <- Matrix(as.matrix(train[varnames]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(test[varnames]), sparse=TRUE)
valid_sparse <- Matrix(as.matrix(valid[varnames]), sparse=TRUE)

# set other labels
listing_id_test <- test$listing_id
labels <- train$class

# Convert data into xgb format
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)
dtest <- xgb.DMatrix(data= test_sparse)
dvalid <- xgb.DMatrix(data = valid_sparse)

# Run cross-validation
xgbPrepropCv <- xgb.cv(data = dtrain,
                       params = param,
                       nrounds = 10000, # change from 50000 
                       maximize=FALSE,
                       prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                       nfold = 5,
                       stratified = TRUE, 
                       print_every_n = 50,
                       early_stopping_round=300)

# Stopping. Best iteration:
#[1250]	train-mlogloss:0.383399+0.003449	test-mlogloss:0.551934+0.010842


###
# Tune other parameters 
###

# grid for tuning
grid <-  expand.grid(
  max_depth = seq(3,5,1), 
  eta = c(0.02), # The range is 0 to 1. Low eta value means model is more robust to overfitting.
  gamma = 1, #  specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be
  colsample_bytree = seq(0.4, 0.5, 0.1 ), 
  min_child_weight = 1, # default 1. 
  subsample = c(0.7 )
)

# create data frame to store tuning history
validationScore <- data.frame(matrix(vector(), nrow(grid), length(grid)+5))
colnames(validationScore) <- c("max_depth", "eta", "gamma","colsample_bytree",
                               "min_child_weight","subsample", "iter", "train_mlogloss_mean",
                               "train_mlogloss_std", "test_mlogloss_mean", "test_mlogloss_std")

for (i in 1:nrow(grid)){
  curr_maxDepth = grid[i, "max_depth"]
  curr_eta = grid[i, "eta"]
  curr_gamma = grid[i, "gamma"]
  curr_colSamplebytree = grid[i, "colsample_bytree"]
  curr_minChild = grid[i, "min_child_weight"]
  curr_subsample = grid[i, "subsample"]

  param <- list(booster="gbtree",
                objective="multi:softprob",
                eval_metric= "mlogloss",
                num_class=3,
                eta = curr_eta,
                gamma = curr_gamma,
                max_depth = curr_maxDepth,
                min_child_weight = curr_minChild,
                subsample = curr_subsample,
                colsample_bytree = curr_colSamplebytree)
  print(paste("starting grid #", i,  Sys.time()))
  modelCV <- xgb.cv(data = dtrain,
                    params = param,
                    nrounds = 5000, # change from 50000 
                    maximize=FALSE,
                    prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                    nfold = 5,
                    stratified = TRUE, 
                    print_every_n = 50,
                    early_stopping_round=300)
  
  # add cv result to table
  score <- cbind(grid[i,],modelCV$evaluation_log[modelCV$best_iteration])
  validationScore[i,] <- score 
}
write.csv(validationScore, file = "analysis/validationScore.csv", row.names = FALSE)

# get best model 
best <- validationScore %>%
  arrange(test_mlogloss_mean) %>%
  top_n(3, -test_mlogloss_mean)
head(best)


# max_depth  eta gamma colsample_bytree min_child_weight subsample iter train_mlogloss_mean train_mlogloss_std test_mlogloss_mean test_mlogloss_std
# 1         4 0.02     1              0.5                1       0.7 1219           0.3834848        0.001762821          0.5526106       0.008425534
# 2         5 0.02     1              0.4                1       0.7  942           0.3472256        0.004842908          0.5542880       0.011497570
# 3         3 0.02     1              0.4                1       0.7 1862           0.4174260        0.001831254          0.5546222       0.013449778

# tune gamma
grid <-  expand.grid(
  max_depth = best$max_depth[1], 
  eta = c(0.02), # The range is 0 to 1. Low eta value means model is more robust to overfitting.
  gamma = c(0.8, 1, 1.2), #  specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be
  colsample_bytree = best$colsample_bytree[1], 
  min_child_weight = best$min_child_weight[1], # default 1. 
  subsample = best$subsample[1]
)

for (i in 1:nrow(grid)){
  curr_maxDepth = grid[i, "max_depth"]
  curr_eta = grid[i, "eta"]
  curr_gamma = grid[i, "gamma"]
  curr_colSamplebytree = grid[i, "colsample_bytree"]
  curr_minChild = grid[i, "min_child_weight"]
  curr_subsample = grid[i, "subsample"]
  
  param <- list(booster="gbtree",
                objective="multi:softprob",
                eval_metric= "mlogloss",
                num_class=3,
                eta = curr_eta,
                gamma = curr_gamma,
                max_depth = curr_maxDepth,
                min_child_weight = curr_minChild,
                subsample = curr_subsample,
                colsample_bytree = curr_colSamplebytree)
  print(paste("starting grid #", i,  Sys.time()))
  modelCV <- xgb.cv(data = dtrain,
                    params = param,
                    nrounds = 5000, # change from 50000 
                    maximize=FALSE,
                    prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                    nfold = 5,
                    stratified = TRUE, 
                    print_every_n = 50,
                    early_stopping_round=300)
  
  # add cv result to table
  score <- cbind(grid[i,],modelCV$evaluation_log[modelCV$best_iteration])
  validationScore <- rbind(validationScore, score)
}

write.csv(validationScore, file = "analysis/validationScore.csv", row.names = FALSE)

# get best model.  
best <- validationScore %>%
  arrange(test_mlogloss_mean) %>%
  top_n(3, -test_mlogloss_mean)
head(best) # Gamma 1 is still good 

# max_depth  eta gamma colsample_bytree min_child_weight subsample iter train_mlogloss_mean train_mlogloss_std test_mlogloss_mean test_mlogloss_std
# 1         4 0.02     1              0.5                1       0.7 1219           0.3834848        0.001762821          0.5526106       0.008425534
# 2         5 0.02     1              0.4                1       0.7  942           0.3472256        0.004842908          0.5542880       0.011497570
# 3         3 0.02     1              0.4                1       0.7 1862           0.4174260        0.001831254          0.5546222       0.013449778

###
# Run models again using Tuned parameters
###

# train model again to confirm performance with validation set
# check variable importance
watchlist <- list(dtrain=dtrain)

param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric= "mlogloss",
              num_class=3,
              eta = best$eta[1],
              gamma = best$gamma[1],
              max_depth = best$max_depth[1],
              min_child_weight = best$min_child_weight[1],
              subsample = best$subsample[1], # use 0.7 instead of 0.9
              colsample_bytree =best$colsample_bytree[1])

model <- xgb.train(data = dtrain,
                  params = param,
                  watchlist=watch,
                  nrounds = best$iter[1],
                  print_every_n = 100
)
predictions <- as.data.table(t(matrix(predict(model, dvalid), nrow=3, ncol=nrow(dvalid))))
colnames(predictions) <- c("low", "medium", "high")
myPred <- as.matrix(predictions)
performance <- MultiLogLoss(y_true = valid$interest_level, y_pred = myPred)
performance #0.4073781??  0.7 subsample.  1287 round


# check variable importance:
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]],model = model)
importance$logGain <- log(importance$Gain)
importance$logCover <- log(importance$Cover)
importance$logFrequency <- log(importance$Frequency)
hist(importance$logGain)
hist(importance$logCover)
hist(importance$logFrequency)
cumSumImportance <- importance %>%
  select(Gain:Frequency) %>%
  mutate_each(funs(cumsum))
colnames(cumSumImportance ) <- paste0("cumSum_",colnames(cumSumImportance ))
importance <- cbind(importance, cumSumImportance )

##
# Model run for Tunned parameters using FULL set of Training data
##

# Run on all available training data
train <- subset(dataTrim, dataTrim$set == "train") # trimmed dataset
train_sparse <- Matrix(as.matrix(train[varnames]), sparse=TRUE)
labels <- train$class
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)

# CV for full training set with tuned parameters to determine nrounds (includes preProp)
# keep same parameters as model 3.  Re-do nrounds CV since there are more rows of data
modelCV <- xgb.cv(data = dtrain,
                  params = param,
                  nrounds = 5000, # change from 50000 
                  maximize=FALSE,
                  prediction = FALSE, #  whether to return the test fold predictions from each CV mode
                  nfold = 5,
                  stratified = TRUE, 
                  print_every_n = 50,
                  early_stopping_round=300)

# [2721]	train-mlogloss:0.362688+0.004236	test-mlogloss:0.504911+0.009200

watchlist <- list(dtrain=dtrain)
xgbFull <- xgb.train(data = dtrain,
                  params = param,
                  #watchlist=watch,
                  nrounds = modelCV$best_ntreelimit)


importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]],model = xgbFull)
xgb.plot.importance(importance_matrix = importance[1:30])


model <- xgbFull
submissionNum <- 13
predTest <- as.data.table(t(matrix(predict(model, dtest), nrow=3, ncol=nrow(dtest))))
colnames(predTest) <- c("low", "medium", "high")
write.csv(data.table(listing_id=listing_id_test, predTest[,list(high,medium,low)]), 
          paste0("submission/submission_xgb_", submissionNum,".csv"), row.names = FALSE)

predTrain <- as.data.table(t(matrix(predict(model, dtrain), nrow=3, ncol=nrow(dtrain))))
colnames(predTrain) <- c("low", "medium", "high")
myPred <- as.matrix(predTrain)
performance <- MultiLogLoss(y_true = train$interest_level, y_pred = myPred)
print(performance) #[1] 0.3862131


###
# Try triming datasets again
###
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]],model = xgbFull)
importance$logGain <- log(importance$Gain)
importance$logCover <- log(importance$Cover)
importance$logFrequency <- log(importance$Frequency)
hist(importance$logGain)
hist(importance$logCover)
hist(importance$logFrequency)
cumSumImportance <- importance %>%
  select(Gain:Frequency) %>%
  mutate_each(funs(cumsum))
colnames(cumSumImportance ) <- paste0("cumSum_",colnames(cumSumImportance ))
importance <- cbind(importance, cumSumImportance )

# Keep cumulative Gain up to 95%. 
discardFeatures <- filter(importance, cumSum_Gain > 0.95)
x <- c("token_", "treat_", "dist_")
discardTokens <- grep(paste(x, collapse = "|"), discardFeatures$Feature, ignore.case = TRUE)
discardTokensName <- discardFeatures[discardTokens, "Feature"]
notDiscarded <- discardFeatures[-discardTokens, "Feature"]

# Trim dataset 
dataTrim2 <- dataTrim[ , -which(names(dataTrim) %in% discardTokensName)]
saveRDS(dataTrim, file = ("data/dataTrim2.Rda"))

###
# Run model on Trimmed Dataset to confirm performance - performance is similar with half the number of features
###
train <- subset(dataTrim2, dataTrim2$frame == "train")

index <- createDataPartition(train$class, 
                             p = .5, 
                             list = FALSE, 
                             times = 1)
train <- train[index,]
valid <- train[-index,]
test <- subset(dataTrim2, dataTrim2$frame == "test")

# pick variables to model
varExclude <- c("order", "listing_id", "set", "frame","interest_level","lowYN", "midYN", "highYN", "class")
varnames <- setdiff(colnames(dataTrim2), varExclude)

# Convert dataset to sparse format
train_sparse <- Matrix(as.matrix(train[varnames]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(test[varnames]), sparse=TRUE)
valid_sparse <- Matrix(as.matrix(valid[varnames]), sparse=TRUE)

# set other labels
listing_id_test <- test$listing_id
labels <- train$class

# Convert data into xgb format
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)
dtest <- xgb.DMatrix(data= test_sparse)
dvalid <- xgb.DMatrix(data = valid_sparse)

# Run cross-validation
param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric= "mlogloss",
              num_class=3,
              eta = .02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .4)


xgbCvTrim2 <- xgb.cv(data = dtrain,
                       params = param,
                       nrounds = 5000, # change from 50000 
                       maximize=FALSE,
                       prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                       nfold = 5,
                       stratified = TRUE, 
                       print_every_n = 50,
                       early_stopping_round=300)

# hope to match Kaggle: 0.55979.  Reduced to 472 variables
# [1210]	train-mlogloss:0.396460+0.003351	test-mlogloss:0.562044+0.008436 New
# [1246]	train-mlogloss:0.380934+0.000940	test-mlogloss:0.558184+0.008779 Old   (difference: [1] 0.00386)

# Run model
watch <- list(dtrain=dtrain)
xgbTrim2Model <- xgb.train(data = dtrain,
                      params = param,
                      watchlist=watch,
                      nrounds = xgbCvTrim2$best_ntreelimit,
                      verbose = 1,
                      print_every_n = 100)

model <- xgbTrim2Model
submissionNum <- 9

predTest <- as.data.table(t(matrix(predict(model, dtest), nrow=3, ncol=nrow(dtest))))
colnames(predTest) <- c("low", "medium", "high")
write.csv(data.table(listing_id=listing_id_test, predTest[,list(high,medium,low)]), 
          paste0("submission/submission_xgb_", submissionNum,".csv"), row.names = FALSE)

predTrain <- as.data.table(t(matrix(predict(model, dtrain), nrow=3, ncol=nrow(dtrain))))
colnames(predTrain) <- c("low", "medium", "high")
myPred <- as.matrix(predTrain)
performance <- MultiLogLoss(y_true = train$interest_level, y_pred = myPred)
print(performance) #[1] 0.8660723.  Investigate why it is so bad

##
#  Re-do grid tuning
##
# grid for tuning
grid <-  expand.grid(
  max_depth = c(4), 
  eta = c(0.02), # The range is 0 to 1. Low eta value means model is more robust to overfitting.
  gamma = 1, #  specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be
  colsample_bytree = seq(0.4, 0.6, 0.1), 
  min_child_weight = 1, # default 1. 
  subsample = c(0.7, 0.8, 0.9)
)

# create data frame to store tuning history
validationScore2 <- data.frame(matrix(vector(), nrow(grid), length(grid)+5))
colnames(validationScore2) <- c("max_depth", "eta", "gamma","colsample_bytree",
                               "min_child_weight","subsample", "iter", "train_mlogloss_mean",
                               "train_mlogloss_std", "test_mlogloss_mean", "test_mlogloss_std")

for (i in 1:nrow(grid)){
  curr_maxDepth = grid[i, "max_depth"]
  curr_eta = grid[i, "eta"]
  curr_gamma = grid[i, "gamma"]
  curr_colSamplebytree = grid[i, "colsample_bytree"]
  curr_minChild = grid[i, "min_child_weight"]
  curr_subsample = grid[i, "subsample"]
  
  param <- list(booster="gbtree",
                objective="multi:softprob",
                eval_metric= "mlogloss",
                num_class=3,
                eta = curr_eta,
                gamma = curr_gamma,
                max_depth = curr_maxDepth,
                min_child_weight = curr_minChild,
                subsample = curr_subsample,
                colsample_bytree = curr_colSamplebytree)
  print(paste("starting grid #", i,  Sys.time()))
  modelCV <- xgb.cv(data = dtrain,
                    params = param,
                    nrounds = 5000, # change from 50000 
                    maximize=FALSE,
                    prediction = TRUE, #  whether to return the test fold predictions from each CV mode
                    nfold = 5,
                    stratified = TRUE, 
                    print_every_n = 50,
                    early_stopping_round=300)
  
  # add cv result to table
  score <- cbind(grid[i,],modelCV$evaluation_log[modelCV$best_iteration])
  validationScore2[i,] <- score 
}
write.csv(validationScore2, file = "analysis/validationScore2.csv", row.names = FALSE)
saveRDS(validationScore2, file = "analysis/validationScore2.Rda" )

# get best model 
best <- validationScore2 %>%
  arrange(test_mlogloss_mean) %>%
  top_n(3, -test_mlogloss_mean)
head(best)
# max_depth  eta gamma colsample_bytree min_child_weight subsample iter train_mlogloss_mean train_mlogloss_std test_mlogloss_mean test_mlogloss_std
# 1         4 0.02     1              0.4                1       0.7 1203           0.3993180        0.006239025          0.5603328       0.009705422
# 2         4 0.02     1              0.4                1       0.9 1232           0.4057378        0.008312902          0.5618836       0.007985805
# 3         4 0.02     1              0.6                1       0.8 1260           0.3898844        0.007937376          0.5619330       0.016499683

# Run on all available training data
train <- subset(dataTrim2, dataTrim2$set == "train") # trimmed dataset
train_sparse <- Matrix(as.matrix(train[varnames]), sparse=TRUE)
labels <- train$class
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)

param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric= "mlogloss",
              num_class=3,
              eta = best$eta[1],
              gamma = best$gamma[1],
              max_depth = best$max_depth[1],
              min_child_weight = best$min_child_weight[1],
              subsample = best$subsample[1], # use 0.7 instead of 0.9
              colsample_bytree =best$colsample_bytree[1])


# CV for full training set with tuned parameters to determine nrounds (includes preProp)
# Re-do nrounds CV since there are more rows of data
modelCV <- xgb.cv(data = dtrain,
                  params = param,
                  nrounds = 5000, # change from 50000 
                  maximize=FALSE,
                  prediction = FALSE, #  whether to return the test fold predictions from each CV mode
                  nfold = 5,
                  stratified = TRUE, 
                  print_every_n = 50,
                  early_stopping_round=300)

# [2425]	train-mlogloss:0.382516+0.002828	test-mlogloss:0.506797+0.003781
watchlist <- list(dtrain=dtrain)
xgbFull2 <- xgb.train(data = dtrain,
                     params = param,
                     watchlist=watch,
                     nrounds = modelCV$best_ntreelimit)
# [2425]	dtrain-mlogloss:0.434479 

# check performance
model <- xgbFull2
submissionNum <- 10
predTest <- as.data.table(t(matrix(predict(model, dtest), nrow=3, ncol=nrow(dtest))))
colnames(predTest) <- c("low", "medium", "high")
write.csv(data.table(listing_id=listing_id_test, predTest[,list(high,medium,low)]), 
          paste0("submission/submission_xgb_", submissionNum,".csv"), row.names = FALSE)

predTrain <- as.data.table(t(matrix(predict(model, dtrain), nrow=3, ncol=nrow(dtrain))))
colnames(predTrain) <- c("low", "medium", "high")
myPred <- as.matrix(predTrain)
performance <- MultiLogLoss(y_true = train$interest_level, y_pred = myPred)
print(performance) # 0.4027963

# check variable importance
# check variable importance:
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]],model = model)
importance$logGain <- log(importance$Gain)
importance$logCover <- log(importance$Cover)
importance$logFrequency <- log(importance$Frequency)
hist(importance$logGain)
hist(importance$logCover)
hist(importance$logFrequency)
cumSumImportance <- importance %>%
  select(Gain:Frequency) %>%
  mutate_each(funs(cumsum))
colnames(cumSumImportance ) <- paste0("cumSum_",colnames(cumSumImportance ))
importance <- cbind(importance, cumSumImportance )

# make confusion matrix (predTrain)  (XGB10)
predTrain$prediction <- colnames(predTrain)[max.col(predTrain[,1:3],ties.method="first")]
predTrain$prediction <- factor(predTrain$prediction, levels = c("low", "medium", "high"))
predTrain$actual <- train$interest_level
predTrain$actual <- factor(predTrain$actual, levels = c("1", "2", "3"))
levels(predTrain$actual) <- list(low = "1", medium = "2", high = "3")
confusion <- table(predTrain$prediction, predTrain$actual) # prediction on left, actual on top
confusion <- as.data.frame(confusion)
names(confusion) <- c("prediction", "actual", "freq")
confusion$percent <- round(100*confusion$freq/nrow(predTrain),2)
confusion

# prediction actual  freq percent
# 1        low    low 32247   65.34
# 2     medium    low  1883    3.82
# 3       high    low   154    0.31
# 4        low medium  4092    8.29
# 5     medium medium  6785   13.75
# 6       high medium   352    0.71
# 7        low   high   569    1.15
# 8     medium   high  1034    2.10
# 9       high   high  2236    4.53

# analyze transactions that are classified as low, even though it should be medium
analysis <- predTrain 
wrongMedium <- filter(analysis, prediction == "low" & actual == "medium" ) 
summary(wrongMedium)

# 
rm(list = ls()[grep("sparse", ls())])
