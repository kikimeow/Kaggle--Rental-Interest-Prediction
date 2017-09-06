# setup
path <- "input path"
setwd(path)

# memory management
memory.limit(size = 100000)

packages <-c("vtreat")  #  load plyr ALWAYS before dplyr, AND not load plyr again.
lapply(packages, FUN = function(X) {do.call("library", list(X))})

data <- readRDS(file = "data/data.Rda")

##
# Run categorical cross-frame experiment on high-cardinality categorical variables
##
train <- subset(data, data$frame == "preProp")

##
# Define function
##
for (level in c("low", "mid", "high")){
  set.seed(1234)
  prep <- vtreat::mkCrossFrameCExperiment(dframe=train, 
                                          varlist=c("display_address", "street_address", "manager_id", "building_id_new"),
                                          outcomename = eval(paste0(level, "YN")), #"interest_midYN"
                                          outcometarget=1, 
                                          rareCount= 1, #optional integer, allow levels with this count or below to be pooled into a shared rare-level.
                                          scale = F, #optional if TRUE replace numeric variables with regression
                                          #minFraction = , #optional minimum frequency a categorical level must have to be converted to an indicator column.
                                          #smFactor = , #optional smoothing factor for impact coding models
                                          #rareSig= , #optional numeric, suppress levels from pooling at this significance value greater. 
                                          ncross = 10)
  # get the "treatment plan" or mapping from original variables to derived variables.
  plan <- prep$treatments
  # get the performance statistics on the derived variables.
  scoreFrame <- plan$scoreFrame
  assign(paste0('scoresFrame_',level),scoreFrame)
  
  # use all the derived variables that have a training significance below 1/NumberOfVariables 
  newVars <- scoreFrame$varName[scoreFrame$sig<1/nrow(scoreFrame)]
  
  # prepare a transformed version of the evaluation/test frame using the treatment plan.
  treatedFrame <- vtreat::prepare(treatmentplan = plan,
                                  dframe = data,
                                  pruneSig=NULL, # suppress variables with significance above this level
                                  varRestriction = newVars)
  colnames(treatedFrame) <- c(paste0("treat_",level, "_", colnames(treatedFrame)))
  assign(paste0('treatedFrame_',level),treatedFrame)
}

# Retaining non-negligible levels as dummy or indicator variables (code: "lev").
# Re-encoding the entire column as an effect code or impact code (code: "catB")
# per-level prevalence (allows pooling of rare or common events) (code: "catP").

# combine dataset
treatedCategories <- cbind(treatedFrame_low, treatedFrame_mid, treatedFrame_high)
treatedCategories <- select(treatedCategories, -treat_low_lowYN, -treat_mid_midYN, -treat_high_highYN)

treatedCategories <- rename(treatedCategories, 
                            treat_mid_manager_id_lev_x = treat_mid_manager_id_lev_x.e6472c7237327dd3903b3d6f6a94515a,
                            treat_low_manager_id_lev_x = treat_low_manager_id_lev_x.e6472c7237327dd3903b3d6f6a94515a)

saveRDS(treatedCategories, file = "data/treatedCategories.Rda")


