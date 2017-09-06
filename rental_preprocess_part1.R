# setup
path <- "input path"
setwd(path)

# memory management
memory.limit(size = 100000)

# create directory
dir.create(paste0("model"))
dir.create(paste0("analysis"))

packages <-c("jsonlite", "tibble", "purrr", "data.table", "dplyr", "caret", "ggmap", "quanteda", "tm", "vtreat")  #  load plyr ALWAYS before dplyr, AND not load plyr again.
lapply(packages, FUN = function(X) {do.call("library", list(X))})

# import files
train <- fromJSON("data/train.json")
test <- fromJSON("data/test.json")

# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names(train), c("photos", "features"))
train <- purrr::map_at(train, vars, unlist) %>% 
  tibble::as_tibble(.)
test <- purrr::map_at(test, vars, unlist) %>% 
  tibble::as_tibble(.)

# make interest_level a factor
train$interest_level <- factor(train$interest_level, levels = c("low", "medium", "high"))
test$interest_level <- NA
test$interest_level <- factor(test$interest_level, levels = c("low", "medium", "high"))

# separate train dataset 30% for other pre-processing, 70% for training
index <- createDataPartition(paste(train$interest_level,
                                   train$manager_id),
                             p = 0.70,
                             list = FALSE)
train$frame <- "train"
train[-index,]$frame <- "preProp" # used for 
train$set <- "train"

# combine test and train set into one for processing
test$frame <- "test"
test$set <- "test"
data <- dplyr:: bind_rows(train, test)

# add a reference column for later sorting
data$order <- seq.int(nrow(data))

data <- select(data, order, listing_id, set, frame, interest_level, price, bathrooms, bedrooms, display_address, 
               street_address, created, manager_id, building_id, features, description, latitude, longitude, photos, everything())

# Convert classes to integers for xgboost
# class <- data.table(interest_level=c("low", "medium", "high"), class=c(0,1,2))
# data <- merge(data, class, by="interest_level", all.x=TRUE, sort=F)

# Add response column to distinguish between low, mid, high
setDT(data)
data[,":="(lowYN =as.integer(interest_level=="low")
          ,midYN=as.integer(interest_level=="medium")
          ,highYN=as.integer(interest_level=="high")
          ,display_address=trimws(tolower(display_address))
          ,street_address=trimws(tolower(street_address)))]

#saveRDS(data, file = "data/data_1.Rda")

#####################
# Cleaning Fields
#####################
# add photo related features
data$numPhotos <- sapply(data$photos, length)
data$numPhotos <- as.numeric(unlist(data$numPhotos))

# remove photos from dataset
data$photos <- NULL

##
# convert features into text string, not list
##
features <- data$features
for (i in 1:length(features)){
  a <- paste0(data$features[i])
  a <- tolower(a)
  a <- gsub("c\\(", a, replacement="")
  a <- gsub("\"", a, replacement= "")
  a <- gsub("\\)", a, replacement= "")
  a <- gsub("list\\(", a, replacement= "")
  features[i] <- a
}
features <- as.data.frame(unlist(features), stringsAsFactors = FALSE )
names(features)[1] <- "featureDes"

# replace column
data$features <- features$featureDes

###
# Summarize Description
###
# get rid of non-ascii characters
data$description <- iconv(data$description, "latin1", "ASCII", sub="")
# remove HTML tags
data$description <- gsub("<.*?>", "", data$description)
# remove "<a  website_redacted "
data$description <- gsub("<a  website_redacted ", "", data$description)
# concatenate feature and description columns
fullDescription <- paste(data$features, data$description, sep = " ")
data$fullDescription <- fullDescription

# re-order
data <- select(data, order:interest_level, lowYN:highYN, price:building_id, numPhotos, latitude:longitude, fullDescription, features, description, everything() )

# Features Column summary
# create corpus (features column)
corpusFeatures <- quanteda::corpus(data$features) # A vector source interprets each element of the vector x as a document.
# feature:  Number of sentences in the combined feature and description
numSentences <- nsentence(corpusFeatures)
data$numSentencesFeatures <- numSentences
# feature: Number of types (unique tokens)
numTypes <- ntype(corpusFeatures, removePunct = TRUE)
data$numTypesFeatures <- numTypes
# feature: count of tokens (total features)
numToken <- ntoken(corpusFeatures, removePunct = TRUE)
data$numTokenFeatures <- numToken

# create corpus (fullDescription column)
corpus <- quanteda::corpus(data$fullDescription) # A vector source interprets each element of the vector x as a document.
summary(corpus, 3, showmeta = TRUE)

# feature:  Number of sentences in the combined feature and description
numSentences <- nsentence(corpus)
data$numSentences <- numSentences

# feature: Number of types (unique tokens)
numTypes <- ntype(corpus, removePunct = TRUE)
data$numTypes <- numTypes

# feature: count of tokens (total features)
numToken <- ntoken(corpus, removePunct = TRUE)
data$numToken <- numToken

saveRDS(data, file = "data/data_2.Rda")

# re-order
data <- select(data, order:longitude, numSentencesFeatures:numToken, fullDescription:description, everything()) 

###
# add date related features
###

data <- data.table::as.data.table(data)
data <- data[, ':='(
  created_year = as.character(substr(created,1,4)), # all in 2016
  created_month = as.character(substr(created,6,7)), # April to June evenly distributed in test & train
  created_day = as.character(substr(created,9,10)),
  created_hh = as.character(substr(created,12,13)),
  created_mm = as.character(substr(created,15,16))
)
]
data$created_mmdd = paste0(data$created_month, data$created_day)
data$created_weekday <- weekdays(as.Date(paste0(data$created_year,"-",data$created_month,"-",data$created_day)))
data$created_weekday <- as.factor(data$created_weekday)
levels(data$created_weekday) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
data.table::setDF(data)

data$created_year <- NULL # all in same year so useless
data$created_month <- as.factor(data$created_month)
data$created_day <- as.factor(data$created_day)
data$created <- as.POSIXct(data$created)
data$created <- as.Date(data$created)
data$created_hh <- as.numeric(data$created_hh)
data$created_mm <- as.numeric(data$created_mm)
data$created_mmdd <- as.factor(data$created_mmdd)

#saveRDS(data, file = "data/data_3.Rda")

###
# add address related features
### 

# fill in missing longitude and latitude data (39 observations)
missingAddress <- data[data$longitude == 0 | data$latitude == 0,]$street_address # create data.frame to store data
outliers_ny <- paste0(missingAddress, ", new york")
missingAddress <- data.frame("street_address" = missingAddress)
coords <- sapply(outliers_ny,
                 function(x) geocode(x, source = "google"))
coords <- t(coords) # transpose
address <- row.names(coords) # just to verify row names agree
missingAddress <- cbind(missingAddress, address, as.data.frame(coords))

data[data$longitude == 0,]$longitude <- missingAddress$lon
data[data$latitude == 0,]$latitude <- missingAddress$lat

data$longitude <- as.numeric(unlist(data$longitude))
data$latitude <- as.numeric(unlist(data$latitude))


# Get neighboorhood coordinates
neighborhood <- c("Downtown Manhattan", "Lower Manhattan", "Upper Manhattan", "Brooklyn", "Queens", "Bronx", "Jersey City", "Staten Island",
                  "Battery Park City", "Bowery", "Chinatown", "Civic Center", "East Village", "Financial District",
                  "Greenwich Village", "Little Italy", "Lower East Side", "NoHo", "NoLita", "SoHo", "Tribeca", "Two Bridges",
                  "West Village", "Chelsea", "Flatiron District", "Garment District", "Gramercy Park", "Hell's Kitchen",
                  "Kips Bay", "Koreatown", "Midtown East", "Murray Hill", "NoMad", "Stuyvesant Town - Peter Cooper Village",
                  "Theater District", "Central Harlem", "Central Park", "East Harlem", "Inwood", "Upper East Side", 
                  "Upper West Side", "Washington Heights", "West Harlem", "Randalls-Wards Island", "Roosevelt Island",
                  "Bedford-Stuyvesant", "Bushwick", "Greenpoint", "Williamsburg", "Boerum Hill", "Carroll Gardens", 
                  "Cobble Hill", "Gowanus", "Greenwood Heights", "Park Slope", "Prospect Park", "Red Hook", "Sunset Park",
                  "Windsor Terrace", "Brooklyn Heights", "Brooklyn Navy Yard", "Clinton Hill", "DUMBO", "Downtown Brooklyn",
                  "Fort Greene", "Prospect Heights", "Vinegar Hill", "Bath Beach", "Bay Ridge", "Bensonhurst", "Borough Park",
                  "Dyker Heights", "Mapleton", "Brownsville", "Canarsie", "Cypress Hills", "East New York", "Bergen Beach",
                  "Flatlands", "Floyd Bennett Airfield", "Marine Park", "Mill Basin", "Astoria", "Corona", "East Elmhurst",
                  "Elmhurst", "Forest Hills", "Glendale", "Jackson Heights", "Long Island City", "Maspeth", "Middle Village",
                  "Rego Park", "Ridgewood","Sunnyside", "Woodside", "Auburndale", "Bayside", "College Point",  "Flushing",
                  "Flushing Meadows-Corona Park", "Fresh Meadows", "Glen Oaks", "Kew Gardens", "Kew Gardens Hills", 
                  "Whitestone", "Briarwood", "Hollis", "Holliswood", "Jamaica", "Jamaica Estates", "Jamaica Hills", "South Jamaica",
                  "St. Albans", "Forest Park", "Howard Beach", "Ozone Park", "Richmond Hill", "South Ozone Park", 
                  "Woodhaven", "Far Rockaway", "Rockaway Beach", "Bedford Park","Bedford Park", "Belmont", "Bronx Park", 
                  "Concourse", "Concourse Village", "East Tremont", "Fordham Heights", "Fordham Manor", "Highbridge", 
                  "Hunts Point", "Kingsbridge", "Longwood", "Marble Hill", "Morris Heights", "Morrisania", "Mott Haven", 
                  "Mount Eden", "Mount Hope", "Norwood", "Riverdale", "University Heights", "Van Cortlandt Park", "West Farms", 
                  "Allerton", "Clason Point", "Morris Park", "Parkchester", "Pelham Bay", "Pelham Parkway", "Throgs Neck", 
                  "Unionport", "Van Nest", "Wakefield", "Westchester Village", "Williamsbridge", "Woodlawn Heights",
                  "Bergen - Lafayette", "Greenville", "Historic Downtown", "McGinley Square", "The Heights",
                  "The Waterfront", "West Side", "East Shore", "Mid-Island", "North Shore", "South Shore"
)
neighborhoods <- paste0(neighborhood, ", New York")
neighborhoodCoords <- sapply(neighborhoods,
                 function(x) geocode(x, source = "google"))
neighborhoodCoords <- t(neighborhoodCoords)
neighborhoodCoords <- as.data.frame(neighborhoodCoords)
row.names(neighborhoodCoords) <- NULL
neighborhoodCoords$lon <- unlist(neighborhoodCoords$lon)
neighborhoodCoords$lat <- unlist(neighborhoodCoords$lat)
neighborhoodCoords$neighborhood <- neighborhood
neighborhoodCoords$neighborhood <- gsub("'", "", neighborhoodCoords$neighborhood)
neighborhoodCoords$neighborhood <- gsub("-", "", neighborhoodCoords$neighborhood)
neighborhoodCoords$neighborhood <- gsub(" ", "", neighborhoodCoords$neighborhood)

# calculate Euclidean distance of property from each neighborhoods
distance <- data.frame(matrix(vector(),nrow(data),nrow(neighborhoodCoords)+1)) #create empty frame
colnames(distance) <- c("listing_id", paste0("dist_",neighborhoodCoords$neighborhood))
distance$listing_id <- data$listing_id
for (i in 1:nrow(neighborhoodCoords)){
  ny_lon <- neighborhoodCoords$lon[i]
  ny_lat <- neighborhoodCoords$lat[i]
  distance[,i+1] <-  mapply(function(lon, lat) sqrt((lon - ny_lon)^2 + (lat - ny_lat)^2),
                            data$longitude, data$latitude)
}
saveRDS(distance, file = "analysis/distance.Rda")

# merge distance dataset with data dataset
data <- dplyr::left_join(data, distance, by = "listing_id")

# list of underground station in New York to flag properties close to public transportation

#saveRDS(data, file = "data/data_4.Rda")

###
# neighborhood related features
###

# get the broader area 
listingNeighborhood <- dplyr::select(data, dist_DowntownManhattan:dist_StatenIsland)
neighborhoodRank <- t(apply(listingNeighborhood, 1, rank, ties.method = "first")) #Rank smallest distance = 1
neighborhoodRank <- as.data.frame(neighborhoodRank)
neighborhood_1 <- apply(neighborhoodRank, 1, function(x) which(x == 1)) #get column number of nearest neighborhood
topNeighborhood <- cbind(neighborhood_1)
topNeighborhood <- as.data.frame(topNeighborhood)

# create look-up table to get neighborhood names
neighborhoodTable <- as.data.frame(cbind(1:170, neighborhood))
colnames(neighborhoodTable) <- c("neighborhoodNum", "neighborhood")
neighborhoodTable$neighborhoodNum <- as.integer(as.character(neighborhoodTable$neighborhoodNum))

topNeighborhood <- dplyr:: left_join(topNeighborhood, neighborhoodTable,  by = c("neighborhood_1" = "neighborhoodNum"))
names(topNeighborhood)[2] <- "location"
location <- as.vector(topNeighborhood$location)
data <- cbind(data,location)

# for each row, get the closest distance to a neighborhood and then assign listing to that neighborhood
listingNeighborhood <- dplyr::select(data, dist_BatteryParkCity:dist_SouthShore)

# get top 5 closest neighborhoods
neighborhoodRank <- t(apply(listingNeighborhood, 1, rank, ties.method = "first")) #Rank smallest distance = 1
neighborhoodRank <- as.data.frame(neighborhoodRank)
neighborhood_1 <- apply(neighborhoodRank, 1, function(x) which(x == 1)) #get column number of nearest neighborhood
neighborhood_2 <- apply(neighborhoodRank, 1, function(x) which(x == 2)) #get column number of nearest neighborhood
neighborhood_3 <- apply(neighborhoodRank, 1, function(x) which(x == 3)) #get column number of nearest neighborhood
neighborhood_4 <- apply(neighborhoodRank, 1, function(x) which(x == 4)) #get column number of nearest neighborhood
neighborhood_5 <- apply(neighborhoodRank, 1, function(x) which(x == 5)) #get column number of nearest neighborhood
top5Neighborhood <- cbind(neighborhood_1, neighborhood_2, neighborhood_3, neighborhood_4, neighborhood_5 )
top5Neighborhood <- as.data.frame(top5Neighborhood)

# # create look-up table to get neighborhood names
# neighborhoodTable <- as.data.frame(cbind(1:55, neighborhood))
# colnames(neighborhoodTable) <- c("neighborhoodNum", "neighborhood")
# neighborhoodTable$neighborhoodNum <- as.integer(as.character(neighborhoodTable$neighborhoodNum))

# append neighborhood to dataset
top5Neighborhood <- dplyr:: left_join(top5Neighborhood, neighborhoodTable,  by = c("neighborhood_1" = "neighborhoodNum"))
top5Neighborhood <- dplyr:: left_join(top5Neighborhood, neighborhoodTable,  by = c("neighborhood_2" = "neighborhoodNum"))
top5Neighborhood <- dplyr:: left_join(top5Neighborhood, neighborhoodTable,  by = c("neighborhood_3" = "neighborhoodNum"))
top5Neighborhood <- dplyr:: left_join(top5Neighborhood, neighborhoodTable,  by = c("neighborhood_4" = "neighborhoodNum"))
top5Neighborhood <- dplyr:: left_join(top5Neighborhood, neighborhoodTable,  by = c("neighborhood_5" = "neighborhoodNum"))
top5Neighborhood <- top5Neighborhood %>%
  dplyr:: select(neighborhood.x:neighborhood) %>%
  rename(neighborhood_1 = neighborhood.x, neighborhood_2 = neighborhood.y, neighborhood_3 = neighborhood.x.x, 
         neighborhood_4 = neighborhood.y.y, neighborhood_5 = neighborhood)

data <- bind_cols(data,top5Neighborhood)

# supply of listing in the neighborhood.  Sum of observation in each neighborhood / total rows
neighborhoodSum <- select(data, neighborhood_1) %>%
  group_by(neighborhood_1) %>%
  summarise(neighborhoodSupply = n()/nrow(data))
data <- left_join(data, neighborhoodSum, by = "neighborhood_1")

rm(vars)
rm(ny_lat)
rm(ny_lon)
rm(outliers_ny)
rm(neighborhood_1, neighborhood_2, neighborhood_3, neighborhood_4, neighborhood_5)
rm(i, neighborhood, neighborhoods, listingNeighborhood, missingAddress, neighborhoodCoords, 
   neighborhoodRank, neighborhoodSum, top5Neighborhood, neighborhoodTable, address, coords, topNeighborhood, location, distance)

#saveRDS(data, file = "data/data_5.Rda")

# save distances to a separate file
distanceFeatures <- select(data, contains("dist_"))
saveRDS(distanceFeatures , file = "data/distanceFeatures.Rda")


###
# bedroom to bathroom ratio
###

# fix listing with 112 bathrooms, change to 1 data$listing_id == "7120577"
data[which(data$bathrooms == 112),]$bathrooms <- 1.5

# fix listing with 20 bathrooms, change to 2 
data[which(data$bathrooms == 20),]$bathrooms <- 2
data[data$bathrooms == 10,]$bathrooms <- 1 # fix data

# bedroom to bathrooms ratio
numBed <- ifelse(data$bedrooms <= 1, 1, data$bedrooms)  # to make studio 1 bedroom for calc
numBath <- ifelse(data$bathrooms <= 1, 1, data$bathrooms)  # to make 0 bathroom 1 for calc
data$bedToBathRatio <- round(numBed/numBath,1)

##
# listing that provided numbers ("phone", "phoneYN",  "areaCodeCount")
##

phone <- stringr:: str_extract_all(data$description, "\\(?\\d{3}\\)?[.-]? *\\d{3}[.-]? *[.-]?\\d{4}")  # List of 124011
for (i in 1:length(phone)){
  a <- paste0(phone[i])
  phone[i] <- a
}
phone <- gsub("character\\(0\\)", phone, replacement="")
phone <- gsub("c\\(", phone, replacement="") # substitute c
phone <- gsub("\\(", phone, replacement="") # substitute (
phone <- gsub("\\)", phone, replacement="") # substitute )
phone <- gsub("\"", phone, replacement= "") # substitute "
phone <- as.data.frame(phone, stringsAsFactors = FALSE )
phone$phoneYN  <- lapply(phone$phone, function(x) ifelse(x == "", 0, 1))
phone$areaCode <- substr(phone$phone, 1, 3)

# append phone to data.  
# for areaCode, replace areaCode by count of area code
areaCode <- phone %>%
  group_by(areaCode) %>%
  summarise(count = n()) %>%
  arrange(-count)
areaCode$percent <- areaCode$count/nrow(areaCode)

data$phoneYN <- unlist(phone$phoneYN)
phone <- phone %>%
  left_join(areaCode, by = "areaCode")

data$areaCodeCount <- phone$count
data$areaCodeCount <- ifelse(data$phoneYN == 0, is.na(data$areaCodeCount) == TRUE, data$areaCodeCount )

##
# listing that provided email
##

email <- stringr:: str_match(data$description, "[[:alnum:]]+\\@[[:alpha:]]+\\.com")  # List of 124011
email <- ifelse(is.na(email), 0, 1)
class(email)
email <- email[,1] 
data$emailYN <- email
rm(email)

###
# Other Miscellaneous
###
# Total number of rooms
numBed <- ifelse(data$bedrooms < 1, 0.5, data$bedrooms)  # to make studio 0.5 bedroom for calc
numBath <- ifelse(data$bathrooms <= 1, 1, data$bathrooms)  # to make 0 bathroom 1 for calc

data$totalRooms <- numBed + numBath

# difference between bedroom and bathroom
data$diffBedBath <- numBed - numBath

# bedroom as % of total room
data$bedPercentTotal <- numBed/data$totalRooms

# display_address formatting
punct <- '[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^]'
data$display_address <- gsub(punct, "", data$display_address)
data$display_address <- trimws(data$display_address, which = c("both"))
data$display_address <- tolower(data$display_address)
display_address <- data$display_address

#saveRDS(data, file = "data/data_6.Rda")

###
# buildingId related features
###

# 20664 building's building id is 0, try to match by street address
buildingIdAddress <- data %>%
  filter(!building_id == "0", !is.na(street_address), !street_address == "") %>%
  select(building_id, street_address) %>%
  group_by(building_id, street_address) %>%
  mutate(ctAddress = n())%>%
  arrange(street_address, ctAddress)%>%
  unique()

# many have same street_address by different building_id, pick the one with most popular building_id.  
# try to find matching street_address
building0 <- subset(data, data$building_id == "0") %>%
  select(listing_id, street_address) %>%
  left_join(buildingIdAddress, by = "street_address") %>%
  arrange(street_address, -ctAddress) %>%
  select(listing_id, building_id, ctAddress)%>%
  group_by(listing_id) %>%
  mutate(ct = n()) %>%
  arrange(listing_id, -ctAddress)

building0  <- data.table(building0, key = "listing_id") # keep first observation of each listing_id 20976
building0 <- building0 [, head(.SD, 1), by = key(building0)] # 20664

# fill in missing building_id  5375 remain missing
data <- left_join(data, select(building0, listing_id:building_id), by = "listing_id")
data$building_id_new <- ifelse(data$building_id.x == "0", data$building_id.y, data$building_id.x)
data <- rename(data, building_id_original = building_id.x)
data$building_id.y <- NULL

# get total number of observations for the building_id including test set
buildingSupplyTotal <- data %>%
  select(building_id_new)%>%
  group_by(building_id_new) %>%
  summarise(buildingIdTotal = n())

data <- left_join(data, buildingSupplyTotal, by = "building_id_new")

#saveRDS(data, file = "data/data_7.Rda")

###
# add manager_id related features
### 
# for each manager_id get the total number of property listings (including test set)
setDT(data)
numPropertyManager <- data[,.(.N), by=manager_id]
data <- dplyr::left_join(data, numPropertyManager, by = "manager_id")%>%
  rename(numPropertyManaged = N)
rm(numPropertyManager)

## measure manager specialization in the building
## group by manager and building and do count
setDT(data)
managerBuildingCount <- data[,.(.N), by = list(manager_id, building_id_new)]
colnames(managerBuildingCount)[3] <- "managerBuildingCount"
data <- left_join(data, managerBuildingCount, by = c("manager_id" = "manager_id", "building_id_new" = "building_id_new"))

## measure manager specilization in the area.
## group by manager and neighborhood 1
setDT(data)
managerNeighborhoodCount <- data[,.(.N), by = list(manager_id, neighborhood_1)]
colnames(managerNeighborhoodCount)[3] <- "managerNeighborhoodCount"
data <- left_join(data, managerNeighborhoodCount, by = c("manager_id" = "manager_id", "neighborhood_1" = "neighborhood_1"))

#saveRDS(data, file = "data/data_8.Rda")

###
# re-Code high cardinality categories using Vtreat
### 
# data <- readRDS(data, file = "data/data_8.Rda")
source(file = "script/rental_vtreat.R")

treatedCategories <- readRDS(file = "data/treatedCategories.Rda")
data <- cbind(data, treatedCategories)

#saveRDS(data, file = "data/data_9.Rda")

##
# Text Analytics:  Features based on "feature" and "description"
##
corpus <- quanteda::corpus(data$fullDescription)
# create tokens used for creating feature matrix
tokensAll<- tokens(char_tolower(corpus), removeNumbers = TRUE, removePunct = TRUE)
tokensNoStopwords <- removeFeatures(tokensAll, stopwords("english"))
tokensNgramsNoStopwords <- tokens_ngrams(tokensNoStopwords, 1:2)
# featnames(dfm(tokensNgramsNoStopwords, verbose = FALSE))

featureMatrix <- dfm(tokensNgramsNoStopwords) # Document-feature matrix of: 124,011 documents, 708,686 features (100% sparse)
topfeatures(featureMatrix, 200)
saveRDS(featureMatrix, file = "data/featureMatrix.Rda")

# keep only words occuring >=10 times and in at least 1% of the documents 
featureMatrixTrim <- dfm_trim(featureMatrix, min_count = 1000, min_docfreq = 0.01)
dim(featureMatrixTrim) # Document-feature matrix of: 124,011 documents, 1,361 features (95.2% sparse)
topFeatures <- topfeatures(featureMatrixTrim, ncol(featureMatrixTrim))
write.csv(topFeatures, file = "analysis/topFeatures.csv")

# coerce dfm to data frame
wordDF <- as.data.frame(featureMatrixTrim, row.names = FALSE)
saveRDS(wordDF, "data/wordDF.Rda")

# make a word cloud
# set.seed(100)
# textplot_wordcloud(featureMatrixTrim , min.freq = 10000, random.order = FALSE,
#                    rot.per = .25, 
#                    colors = RColorBrewer::brewer.pal(8,"Dark2"))

# to check fo individual terms
# kwic(corpus, "natural sunlight")
# kwic(corpus, "king")

# custom features: terms that capture majority of features
x <- c("central ac", "central a/c", "central air", "a/c", "air condition", "climate control")
data$des_AC <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("balcony", "balconies")
data$des_balcony <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("bike room", "bicycle storage", "bike storage", "bicycle storage")
data$des_bike <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("business center")
data$des_businessCenter <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("conference room")
data$des_conferenceRoom <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("cinema", "movie", "private screening", "screening room")
data$des_cinema <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("cable", "satellite")
data$des_cable <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("live in super", "concierge", "superintendent", "valet", "live-in super", "on-site super", "hotel service", "hotel-service")
data$des_concierge <- grepl(paste(x, collapse = "|"), data$fullDescription) 

x <- c("full service", "full-service")
data$des_fullService <- grepl(paste(x, collapse = "|"), data$fullDescription)

x <- c("dining")
data$des_dining <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("dish washer", "dishwasher")
data$des_dishwasher <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("microwave")
data$des_microwave <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("simplex")
data$des_simplex <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("duplex")
data$des_duplex <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("triplex")
data$des_triplex <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("elevator", "elev bldg")
data$des_elevator <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("exposed brick", "exposedbrick", "exposed-brick", "brick walls")
data$des_brick <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("fire place", "fireplace")
data$des_fireplace <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("furnished", "furniture include")
data$des_furnished <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("garage", "parking")
data$des_parking <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("green building")
data$des_green <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("fitness", "gym", "health club", "exercise", "yoga", "dance", "weight training", "basketball")
data$des_gym <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("hardwood", "hard wood", "wooden floors", "wood floor", "hard-wood", "wood-floor")
data$des_hardwood <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("speed internet", "wireless internet", "wifi")
data$des_internet <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("high ceiling", "high-ceiling", "highceiling")
data$des_highceiling <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("hi rise", "highrise")
data$des_highrise <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("lowrise")
data$des_lowrise <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("midrise")
data$des_midrise <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("laundry", "dryer", "washer and dryer", "washer dryer", "washer & dryer", "washer/dryer")
data$des_laundry <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("dry clean", "dry-cleaning", "dry-clean")
data$des_dryclean <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("natural light", "sunlight", "bright", "sunny")
data$des_light <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("loft")
data$des_loft <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("lounge")
data$des_lounge <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("marble")
data$des_marble <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("multilevel", "multi-level", "multi level")
data$des_multilevel <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("new construction", "new building", "new-building", "new-construction")
data$des_newConstruction <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("new appliance", "new stainless")
data$des_newAppliance <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("no fee", "no-fee", "no broker fee")
data$des_noFee <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("reduced fee", "reduced-fee")
data$des_reducedFee <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("roof deck", "outdoor", "terrace", "garden", "roof-deck", "courtyard", "backyard", "deck", "patio")
data$des_outdoor <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("penthouse")
data$des_penthouse <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("no pet")
data$des_noPets <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("pets", "cats allowed", "dogs allowed", "pet")
pets <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 
data$des_petsOkay <- ifelse(data$des_noPets  == TRUE, FALSE, pets )

x <- c("playroom", "play room", "nursery")
data$des_playroom <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("granite")
data$des_granite <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("pool")
data$des_pool <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("swimming pool") # to filter out pool table
data$des_swimmingPool <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("post war", "postwar", "post-war")
data$des_postwar <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("pre war", "prewar", "pre-war")
data$des_prewar <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("renovated", "updated kitchen and bathroom", "updated kitchen", "renovation", "new kitchen", "new bath", "new bathroom")
data$des_renovated <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("sauna")
data$des_sauna <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("short term", "short-term")
data$des_shortTerm <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("sky light", "skylight", "sky-light")
data$des_skylight <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("stainless")
data$des_stainless <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("storage")
data$des_storage <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("sublet")
data$des_sublet <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("subway", "train", "transportation")
data$des_subway <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("park[[:space:]]", "parks[[:space:]]")  
data$des_park <- grepl(pattern = paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("walk in closet", "walk-in closet")
data$des_walkincloset <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("wheelchair")
data$des_wheelchair <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("floor to ceiling", "floor-to-ceiling", "floortoceiling")
data$des_floorToCeiling <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("barbecue", "bbq")  
data$des_barbecue <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("attended lobby", "doorman", "door man")  
data$des_doorman <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("custom cabinetry")  
data$des_cabinetry <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("modern kitchen")  
data$des_modernKitchen <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("open kitchen")  
data$des_openKitchen <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("oversized windows", "enormous widows", "large window", "huge window")  
data$des_oversizedWindows <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("window")  
data$des_windows <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("recessed light", "recess light", "recess custom light")  
data$des_recessedLighting <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("soaking tub")  
data$des_soakingTub <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("tub")  
data$des_tub <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("tiled bathroom")  
data$des_tiledBathroom <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("utilities included")  
data$des_utilitiesIncluded <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("water included")  
data$des_waterIncluded <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("view of", "views of", "amazing view")
data$des_viewOf <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("windowed kitchen")  
data$des_windowedKitchen <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("wood cabinets")  
data$des_woodCabinets <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("king size", "king bed", "king-size", "king or queen", "king/queen")
data$des_kingSize <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("queen size", "queen bed", "queen-size")
data$des_queenSize <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("building laundry")
data$des_buildingLaundry <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("floor throughout")
data$des_floorThroughout <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("month free", "months free")
data$des_monthFree <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("separate kitchen")
data$des_separateKitchen <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE) 

x <- c("closet")
data$des_closet <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("24 hour", "24-hour", "24 hr", "24-hr")
data$des_24Hour <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("central park")
data$des_centralPark <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

x <- c("manhattan")
data$des_manhattan <- grepl(paste(x, collapse = "|"), data$fullDescription, ignore.case = TRUE)

# change from logical to numeric
logicals <- sapply(data, is.logical)
data[,logicals] <- as.numeric(unlist(data[,logicals]))

#saveRDS(data, file = "data/data_10.Rda")
saveRDS(data, file = "data/data.Rda")


# run model to obtain predicted medium price using the rental_model_price.R file




