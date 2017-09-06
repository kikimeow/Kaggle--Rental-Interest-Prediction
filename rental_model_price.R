# set your working directory
path <- "input path"
setwd(path)

# load packages
packages <-c("dplyr", "caret", "data.table")  #  load plyr ALWAYS before dplyr, AND not load plyr again.

lapply(packages, FUN = function(X) {
  do.call("library", list(X)) 
})

# memory management
memory.limit(size = 100000)

###
# Predict price based on subsets of "Medium+High", "Medium", "All"
###

# load files
data <- readRDS(file = "data/data.Rda")

# Fill in column to be ignored by model.  To get rid of list in creating H2O frames
colExclude <- c("created", "description", "features", "photos", "street_address", "fullDescription", "display_address")
data <-  data[, !(colnames(data) %in% colExclude)]

# excludes outliers in the training set
quantile(data$price, c(.75, .90, .95, .975, .98, .99, .995, .999, .9999))
# 75%      90%      95%      97.5%      98%      99%    99.5%    99.9%   99.99% 
# 4100.0   5650.0   6875.0   8800.0   9867.4  13000.0  16000.0  29995.0 125420.5 

quantile(data$price, c(.00003, .00005, .0001, .0005, 0.01, 0.05, 0.1))
# 0.003%    0.005%     0.01%     0.05%        1%        5%       10% 
# 386.7353  539.0975  710.0250 1050.0000 1495.0000 1800.0000 2000.0000 

df <- subset(data, price < quantile(data$price, c(.99))[[1]]) # exclude price higher than 99% percentile
df <- subset(df, price > quantile(data$price, c(.01))[[1]]) # exclude price lower than 1%  percentile
df <- subset(df, frame == "preProp")
preProp <- df

# setup environment
library(h2o)  # Requires version >=0.0.4 of h2oEnsemble
localH2O <- h2o.init(nthreads = -1, max_mem_size = "12G")  # Start an H2O cluster with nthreads = num cores on your machine.  Use all cores available to you (-1)
h2o.removeAll() # Clean slate - just in case the cluster was already running

# Setup X (predictors) & Y (response)
treat <- grep("treat_", names(data), ignore.case = TRUE)
created <- grep("created", names(data), ignore.case = TRUE)
colExclude <- c("order", "frame", "manager_id", "lowYN", "midYN", "highYN", "building_id_original" , "interest_level", 
                "set", "numPhotos", "skill_low", "skill_medium",  "skill_high","building_id_new", "managerBuildingCount",
                "managerNeighborhoodCount", 
                "managerSkill", "numPropertyManaged", "buildingInterest_low", "buildingInterest_medium", "buildingInterest_high",
                "buildingInterest", "buildingIdTotal", "phoneYN", "areaCodeCount",  "emailYN", "numSentencesFeatures",
                "numTypesFeatures", "numTokenFeatures" ,  "numSentences", "numTypes", "numToken" , "displayAddressCount",
                "display_address", "listing_id",
                colnames(data[treat]), 
                colnames(data[created]))

response <- "price"
predictors <- setdiff(names(df),c(response, colExclude) )  # need to exclude the rest of added.prod
predictors

# make h2o frames frames
df <- as.h2o(df, destination_frame = "df.hex") # 122673 observations
all_data <- as.h2o(data, destination_frame = "all_data.hex") # include filtered observations by quantiles. Need this for predictions

# train on 90% of data, save 10% data for validation
splits <- h2o.splitFrame(
  data = df,
  ratios = c(0.9),
  destination_frames = c("train.hex", "valid.hex"), seed = 11121)
train <- splits[[1]]
valid <- splits[[2]]

##
# Train on ALL interests
##

# Grid Search for RandomForest
hyper_params = list( 
  max_depth = c(20, 30, 40), # default 20
  #min_rows = c(1, 2^seq(0,log2(nrow(train))-1,1)[3:9]), # default 1  (this is 1 to 256)
  nbins = 2^seq(4,10,1)[1:3], #default 20 this:16 32 64
  #nbins_cats = 2^seq(4,10,1), #default 1024 this: 16   32   64  128  256  512 1024
  mtries =  c(round(length(predictors)/seq(2,4,1)))#, #defaults to sqrtp for classification and p/3 for regression (where p is the # of predictors Defaults to -1. Number of variables randomly sampled as candidates at each split. 
  #sample_rate = seq(0.5,0.9,0.15), # Defaults to 0.6320000291
  #min_split_improvement = c(1e-6,1e-5,1e-4) #Defaults to 1e-05
)
hyper_params

search_criteria = list(
  strategy = "RandomDiscrete",      
  max_models = 10,                  
  seed = 1131,
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 3,                
  stopping_metric = "AUTO",
  stopping_tolerance = 0.001
)

grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "randomForest",
  grid_id = "randomForestAllPrice", 
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ntrees = 1000, # default is 50                                                           
  
  ## early stopping once the validation measurement doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, 
  stopping_tolerance = 0.0001, 
  stopping_metric = "AUTO", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)
grid

# 
# max_depth	mtries	nbins	model_ids	residual_deviance
# 30	91	32	randomForestAllPrice_model_7	450588.41296460206
# 40	68	16	randomForestAllPrice_model_2	450704.8088717737
# 20	68	16	randomForestAllPrice_model_0	451771.40395161463

# We can inspect the best 3 models from the grid search explicitly.
for (i in 1:3){
  print(grid@model_ids[[i]])
  model <- h2o.getModel(grid@model_ids[[i]])
  print(h2o.mse(h2o.performance(model, valid = TRUE)))
  print(h2o.mse(h2o.performance(model, newdata = all_data)))
  model_path <- h2o.saveModel(model,  paste0(getwd(), "/model/"), force = TRUE)
  print(model_path)}

winner <- h2o.getModel(grid@model_ids[[1]]) #C:\\Users\\kikimeow\\Documents\\Kaggle\\Kaggle- Rentals\\model\\randomForestAllPrice_model_9
winner@parameters


# make predictions using model
predictH2O <- predict(winner, all_data)
head(predictH2O)
predictions <- as.vector(predictH2O)
head(predictions)
predictionsAll <- predictions

# save the predicted price
saveRDS(predictions, file = "data/AllPricePrediction.Rda")

##
#  Predict prices with only medium and high interest
##
df <- preProp
df <- subset(df, interest_level != "low") # exclude low interest entries.  
df <- as.h2o(df, destination_frame = "df.hex") 

# re-do grid search, this time with 5 fold validation since there's less data
grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "randomForest",
  grid_id = "randomForestMidHighPrice", 
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = df, 
  #validation_frame = valid,
  nfolds = 5, 
  ntrees = 1000, # default is 50.  Use early stopping so use higher ntrees                                                           
  
  ## early stopping once the validation measurement doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, 
  stopping_tolerance = 0.0001, 
  stopping_metric = "AUTO", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 123                                                             
)
grid

# Hyper-Parameter Search Summary: ordered by increasing residual_deviance
# max_depth mtries nbins                         model_ids  residual_deviance
# 1         20     68    16 randomForestMidHighPrice2_model_0 241445.08401483303
# 2         40     68    16 randomForestMidHighPrice2_model_2  241514.2457546774
# 3         40     68    32 randomForestMidHighPrice2_model_1 242237.86987463248


# We can inspect the best 3 models from the grid search explicitly.
for (i in 1:3){
  print(grid@model_ids[[i]])
  model <- h2o.getModel(grid@model_ids[[i]])
  print(h2o.mse(h2o.performance(model, newdata = all_data)))
  model_path <- h2o.saveModel(model,  paste0(getwd(), "/model/"), force = TRUE)
  print(model_path)}

# make predictions using model
winner <- h2o.getModel(grid@model_ids[[1]])
predictH2O <- predict(winner, all_data)
head(predictH2O)
predictions <- as.vector(predictH2O)
head(predictions)
predictionsMidHigh <- predictions

# save the predicted price
saveRDS(predictions, file = "data/mediumHighPricePrediction.Rda")

##
# Use same model to predict prices with medium interest only
##
df <- preProp
df <- subset(df, interest_level == "medium") 
df <- as.h2o(df, destination_frame = "df.hex") 

# run with same parameter as the winning model (randomForestAllPrice_model_9)
# re-do grid search, this time with 5 fold validation since there's less data
grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "randomForest",
  grid_id = "randomForestMediumPrice", 
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = df, 
  #validation_frame = valid,
  nfolds = 5, 
  ntrees = 1000, # default is 50.  Use early stopping so use higher ntrees                                                           
  
  ## early stopping once the validation measurement doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 3, 
  stopping_tolerance = 0.0001, 
  stopping_metric = "AUTO", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 124                                                             
)
grid
# max_depth mtries nbins                       model_ids  residual_deviance
# 1         30     91    32 randomForestMediumPrice_model_7  274318.5522507534
# 2         40     68    32 randomForestMediumPrice_model_1 275142.34086581244
# 3         20     91    64 randomForestMediumPrice_model_3 276362.54317784443

winner <- h2o.getModel(grid@model_ids[[1]])
model_path <- h2o.saveModel(winner,  paste0(getwd(), "/model/"), force = TRUE)

# make predictions using model
predictH2O <- predict(winner, all_data)
head(predictH2O)
predictions <- as.vector(predictH2O)
head(predictions)
predictionsMid <- predictions

# save the predicted price
saveRDS(predictions, file = "data/mediumPricePrediction.Rda")

# save all prices in one file
predictions <- cbind(predictionsAll, predictionsMidHigh, predictionsMid)
predictions <- as.data.frame(predictions)
saveRDS(predictions, file = "data/PricePredictionAll3.Rda")

# other notes
# fold_assignment = "Stratified"
# balance_classes = TRUE,
# class_sampling_factors = c(1, 1, 2.5)


