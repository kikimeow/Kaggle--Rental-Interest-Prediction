# set your working directory
path <- "input path"
setwd(path)

# load packages
packages <-c("dplyr", "caret", "data.table", "xgboost")  #  load plyr ALWAYS before dplyr, AND not load plyr again.

lapply(packages, FUN = function(X) {
  do.call("library", list(X)) 
})

# memory management
memory.limit(size = 100000)

###
# Predict price based on subsets of "Medium+High", "Medium"
###

# load files
data <- readRDS(file = "data/data.Rda")

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


# excludes outliers in the training set
df <- subset(data, price < quantile(data$price, c(.99))[[1]]) # exclude price higher than 99% percentile
df <- subset(df, price > quantile(data$price, c(.01))[[1]]) # exclude price lower than 1%  percentile
df <- subset(df, frame == "preProp")
preProp <- df


# Setup X (predictors) & Y (response)
treat <- grep("treat_", names(data), ignore.case = TRUE)
created <- grep("created", names(data), ignore.case = TRUE)
varExclude <- c("order", "frame", "manager_id", "lowYN", "midYN", "highYN", "building_id_original" , "interest_level", 
                "set", "numPhotos", "skill_low", "skill_medium",  "skill_high","building_id_new", "managerBuildingCount",
                "managerNeighborhoodCount", 
                "managerSkill", "numPropertyManaged", "buildingInterest_low", "buildingInterest_medium", "buildingInterest_high",
                "buildingInterest", "buildingIdTotal", "phoneYN", "areaCodeCount",  "emailYN", "numSentencesFeatures",
                "numTypesFeatures", "numTokenFeatures" ,  "numSentences", "numTypes", "numToken" , "displayAddressCount",
                "display_address", "listing_id",
                colnames(data[treat]), 
                colnames(data[created]),
                "class")
varnames <- setdiff(colnames(preProp), varExclude)  # need to exclude the rest of added.prod
varnames

##
# Keep observations for mid and high only
##
df <- df[df$interest_level != "low",]

# Convert dataset to sparse format
train_sparse <- Matrix(as.matrix(df[varnames]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(data[varnames]), sparse=TRUE)

# set other labels
listing_id_test <- data$listing_id
labels <- df$price

# Convert data into xgb format
dtrain <- xgb.DMatrix(data= train_sparse, label=labels)
dtest <- xgb.DMatrix(data= test_sparse)

##
# Tuning
##
grid <-  expand.grid(
  max_depth = c(3), 
  eta = c(0.02), # The range is 0 to 1. Low eta value means model is more robust to overfitting.
  gamma = 1, #  specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be
  colsample_bytree = seq(0.6, 0.8, 0.2), 
  min_child_weight = 1, # default 1. 
  subsample = c(0.5, 0.7, 0.9)
)

# create data frame to store tuning history
validationScore <- data.frame(matrix(vector(), nrow(grid), length(grid)+5))
colnames(validationScore) <- c("max_depth", "eta", "gamma","colsample_bytree",
                               "min_child_weight","subsample", "iter", "train_rmse_mean",
                               "train_rmse_std", "test_rmse_mean", "test_rmse_std")

for (i in 1:nrow(grid)){
  curr_maxDepth = grid[i, "max_depth"]
  curr_eta = grid[i, "eta"]
  curr_gamma = grid[i, "gamma"]
  curr_colSamplebytree = grid[i, "colsample_bytree"]
  curr_minChild = grid[i, "min_child_weight"]
  curr_subsample = grid[i, "subsample"]
  
  param <- list(booster="gbtree",
                objective="reg:linear",
                eval_metric= "rmse",
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
                    nfold = 5,
                    print_every_n = 100,
                    early_stopping_round=300)
  
  # add cv result to table
  score <- cbind(grid[i,],modelCV$evaluation_log[modelCV$best_iteration])
  validationScore[5+i,] <- score 
}
write.csv(validationScore, file = "analysis/validationScoreRentalPriceXGB.csv", row.names = FALSE)

# get best model 
best <- validationScore %>%
  arrange(test_rmse_mean) %>%
  top_n(3, -test_rmse_mean)
head(best)

# max_depth  eta gamma colsample_bytree min_child_weight subsample iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
# 1         3 0.02     1              0.8                1       0.5 4999        7.101394      1.2004268       14.65064      1.496060
# 2         3 0.02     1              0.8                1       0.7 5000        7.137306      1.3104760       15.48484      2.883844
# 3         3 0.02     1              0.8                1       0.9 5000        7.785322      0.4453389       17.74963      3.790980


# Run cv with tuned parameters
watch <- list(dtrain=dtrain)
param <- list(booster="gbtree",
              objective="reg:linear",
              eval_metric= "rmse",
              eta = best$eta[1],
              gamma = best$gamma[1],
              max_depth = best$max_depth[1],
              min_child_weight = best$min_child_weight[1],
              subsample = best$subsample[1], 
              colsample_bytree =best$colsample_bytree[1])

model <- xgb.train(data = dtrain,
                    params = param,
                    watchlist=watch,
                    nrounds = best$iter[1]
)

# Predictions
predictions <- as.data.table(t(matrix(predict(model, dtest), nrow=1, ncol=nrow(dtest))))
colnames(predictions) <- c("predPrice_xgb")

saveRDS(predictions, "data/priceXGB.Rda")
xgbPrice <- xgb.save.raw(model)
saveRDS(xgbPrice , "data/xgbPrice.Rda")

# see variable importance
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]], model = model)
importance[1:100]


