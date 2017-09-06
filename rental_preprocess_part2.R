# setup
path <- "input path"
setwd(path)

packages <-c("dplyr", "caret") #  load plyr ALWAYS before dplyr, AND not load plyr again.
lapply(packages, FUN = function(X) {do.call("library", list(X))})

# memory management
memory.limit(size = 100000)


# read data
data <- readRDS(file = "data/data.Rda")
#data <- readRDS(file = "data/data_10.Rda")
PricePredictionAll3 <- readRDS(file = "data/PricePredictionAll3.Rda")
PricePredictionXGB <- readRDS(file = "data/priceXGB.Rda")
wordDF <- readRDS(file = "data/wordDF.Rda") #1361 variables

###
# Price features
###
setDF(data)
##
# From Random Forest
##
# add predicted price based on medium interest to data
data$pricePredAll <- PricePredictionAll3$predictionsAll
data$pricePredMidHigh <- PricePredictionAll3$predictionsMidHigh
data$pricePredMedium <- PricePredictionAll3$predictionsMid
data$pricePredAvg <- rowMeans(subset(data, select = c(pricePredAll , pricePredMidHigh, pricePredMedium)))

# priceDiffMedium = price - predicted price.  (+ = too expensive, - = cheap)
data$priceDiffAll <- round((data$price - data$pricePredAll),0)
data$priceDiffMidHigh <- round((data$price - data$pricePredMidHigh),0)
data$priceDiffMedium <- round((data$price - data$pricePredMedium),0)
data$priceDiffAvg<- round((data$price - data$pricePredAvg),0)

# percentDiffMedium = (price - predicted price) / price
data$percentDiffAll <- round((data$price - data$pricePredAll)/data$price,4)
data$percentDiffMidHigh <- round((data$price - data$pricePredMidHigh)/data$price,4)
data$percentDiffMedium <- round((data$price - data$pricePredMedium)/data$price,4)
data$percentDiffAvg <- round((data$price - data$pricePredAvg)/data$price,4)

##
# From xgboost
##
# add predicted price based on medium interest to data
data$pricePredMidHighXgb <- PricePredictionXGB$predPrice_xgb

# priceDiffMedium = price - predicted price.  (+ = too expensive, - = cheap)
data$priceDiffMidHighXgb <- round((data$price - data$pricePredMidHighXgb),0)

# percentDiffMedium = (price - predicted price) / price
data$percentDiffMidHighXgb <- round((data$price - data$pricePredMidHighXgb)/data$price,4)

# price per bedroom
numBed <- ifelse(data$bedrooms < 1, 0.5, data$bedrooms)  # to make studio 0.5 bedroom for calc
data$pricePerBedroom <- data$price/numBed

# price per bathroom
numBath <- ifelse(data$bathrooms <= 1, 1, data$bathrooms)  # to make 0 bathroom 1 for calc
data$pricePerBathroom <- data$price/numBath

# Price per Room
numBed <- ifelse(data$bedrooms < 1, 0.5, data$bedrooms)  # to make studio 0.5 bedroom for calc
numBath <- ifelse(data$bathrooms <= 1, 1, data$bathrooms)  # to make 0 bathroom 1 for calc
data$pricePerRoom <- data$price/(numBed+numBath)


###
# Difference from median price of neighborhood depending the number of bedrooms.  
# Bedrooms >=5 count as 4
# Take average of 1 to 5 neighborhoods 
###
neighborhoodPrice <- data %>%
  mutate(newNumBed = ifelse(bedrooms >= 4, 4, bedrooms)) %>%
  select(neighborhood_1, price, newNumBed )%>%
  group_by(neighborhood_1, newNumBed)%>%
  summarise(numProp = n(), averagePrice = mean(price), medianPrice = median(price), sdPrice = sd(price), minPrice = min(price), maxPrice = max(price))

neighborPriceData <- data %>%
  mutate(newNumBed = ifelse(bedrooms >= 4, 4, bedrooms)) %>%
  select(neighborhood_1:neighborhood_5, price, newNumBed)%>%
  left_join(select(neighborhoodPrice, neighborhood_1, newNumBed,  medianPrice), 
            by = c("neighborhood_1" = "neighborhood_1", "newNumBed" = "newNumBed"))%>%
  left_join(select(neighborhoodPrice, neighborhood_1, newNumBed,  medianPrice), 
            by = c("neighborhood_2" = "neighborhood_1", "newNumBed" = "newNumBed"))%>%
  left_join(select(neighborhoodPrice, neighborhood_1, newNumBed,  medianPrice), 
            by = c("neighborhood_3" = "neighborhood_1", "newNumBed" = "newNumBed"))%>%
  left_join(select(neighborhoodPrice, neighborhood_1, newNumBed,  medianPrice), 
            by = c("neighborhood_4" = "neighborhood_1", "newNumBed" = "newNumBed"))%>%
  left_join(select(neighborhoodPrice, neighborhood_1, newNumBed,  medianPrice), 
            by = c("neighborhood_5" = "neighborhood_1", "newNumBed" = "newNumBed"))

neighborPriceData <- neighborPriceData %>%
  mutate(mean1 = medianPrice.x)%>%
  mutate(mean2 = (medianPrice.x+medianPrice.y)/2) %>%
  mutate(mean3 = (medianPrice.x+medianPrice.y+medianPrice.x.x)/3) %>%
  mutate(mean4 = (medianPrice.x+medianPrice.y+medianPrice.x.x+medianPrice.y.y)/4) %>%
  mutate(mean5 = (medianPrice.x+medianPrice.y+medianPrice.x.x+medianPrice.y.y+medianPrice)/5)

neighborPriceData <- neighborPriceData %>%
  mutate(diffPrice_neighbor1 = price - mean1)%>%
  mutate(diffPrice_neighbor2 = price - mean2)%>%
  mutate(diffPrice_neighbor3 = price - mean3)%>%
  mutate(diffPrice_neighbor4 = price - mean4)%>%
  mutate(diffPrice_neighbor5 = price - mean5)

neighborPriceData <- neighborPriceData %>%
  mutate(diffPercent_neighbor1 = (price - mean1)/price)%>%
  mutate(diffPercent_neighbor2 = (price - mean2)/price)%>%
  mutate(diffPercent_neighbor3 = (price - mean3)/price)%>%
  mutate(diffPercent_neighbor4 = (price - mean4)/price)%>%
  mutate(diffPercent_neighbor5 = (price - mean5)/price)

data <- bind_cols(data, select(neighborPriceData, diffPrice_neighbor1:diffPercent_neighbor5))

#saveRDS(data, file = "data/data_11.Rda")

###
# add wordDF to data
###
colnames(wordDF) <- paste("token", colnames(wordDF), sep = "_")
data <- dplyr::bind_cols(data, wordDF)

#saveRDS(data, file = "data/data_12.Rda")
saveRDS(data, file = "data/data.Rda")

