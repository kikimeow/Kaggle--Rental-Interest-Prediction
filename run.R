# **************************************
# Setup
# **************************************

# set your working directory
path <- "input path here"
setwd(path)

# create directory
dir.create(paste0("data")) # Please put the download data into this folder
dir.create(paste0("model output"))
#dir.create(paste0("script"))

# **************************************
# run scripts
# **************************************

source("script/rental_preprocess_part1.R")
source("script/rental_model_price.R")
source("script/rental_model_price_xgb.R")
source("script/rental_preprocess_part2.R")
source("script/rental_model_classification")
