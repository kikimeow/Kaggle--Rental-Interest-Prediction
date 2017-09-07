## Two Sigma Connect: Rental Listing Inquiries

### Overview
The Rental Listing Inquiries is a multi-class classification competition hosted by [Kaggle](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries).  The goal of the project is to predict the popularity of an apartment rental listing given the listing's attributes, such as text description, photos, number of bedrooms, price, etc. The data comes from [renthop.com](https://www.renthop.com/), an apartment listing website.  The apartments in the dataset are located in the New York City.  The target variable "interest_level" has 3 categories: high', 'medium', 'low'.

Some of the challenges faced in the project involves text analytics, working with high-cardinality categorical features, and mapping.

The training set has 49352 observations, and the testing set has 74659 observations.
The data fields given are as follows:
* bathrooms: number of bathrooms
* bedrooms: number of bathrooms
* building_id
* created: date and timestamp of when the listing was created
* description
* display_address
* features: a list of features about this apartment
* latitude
* listing_id
* longitude
* manager_id
* photos: a list of photo links
* price: in USD
* street_address
* interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'

### Instruction to run the files
Create two file folders "/script" and "/data". Save the R scripts to script folder, and the data from Kaggle to the data folder.
Each of the R script has the variable "Path".   Modify the path in the scripts.  The run.R file will run each of the scripts.

### Process Overview
One key component of the model is to predict the fair-price of a listing prior to predicting the interest level of the property.  To calculate fair-price, two models are run- the first is Random Forest model using H2O, and the second is XGBoost.  XGBoost is used again to predict the interest level of the listing.  

### Feature Engineering
Below are the reasonings behind some of the new features added to the modeling process:  

**Neighborhood** is an important factor in determining the attractiveness of a real estate.   A list of neighborhood names in New York was compiled.  The Google Maps Geocoding API is then used to find the coordinates of the neighborhoods.  Euclidian distance is used to measure the distance of the property to the various neighborhoods.  The new features added includes the nearest 5 neighborhoods.  The neighborhoods are also used to derive comparable prices in the same neighboorhoods.   

**Building_id** is one of the features that added value in the model.  The intuition is that if a building is attractive, most of the units from the building will be attractive as well.  To capture the desirability of each building, building_id is re-encoded by the probability of each interest_level, conditional on each building_id.  For example, the new variable could be the probability of interest_level being high for the a particular building_id.  The [vtreat](https://arxiv.org/pdf/1611.09477.pdf) package is used to encode the "building_id". Cross-validation is used to avoid data leakage in treating the variable.

The count of the Building_id was also added as a new feature since interest_level might be affected by supply of the building.  

**Manager_id** is the other high-cardinality feature.  Property management is a skill.  The skill of the property manager affects the desirability of the property.  To capture manager skill, similarly to Building_ID, the vtreat package was used to encode Manager_ID with respect to interest_level.  

In addition, features are also added to capture Manager’s specialization in a building, and in the different neighborhoods.  The assumption is that if a Manager has many listings in a building or a neighborhood, the specialization may make the listing more desirable.  

There is a separate feature to capture manager_id that only has one listing.  These group of people might be the mom-and-pops that has a single property that they manage themselves.  

**Description** and **Feature** are the two text fields in the dataset.  The two fields are combined to a single corpus.  The most frequent keywords of the properties are added as new features.  For example, some of keywords are: central AC, balcony, subway, doorman, view, central park, Manhattan etc.  Words of similar meanings are grouped together.  The count of words, and count of sentences were also added since descriptive explanations may spur the interest of a property.  

**Price** is normally the most important factor in determining the desirability of a product.  To measure the value proposition of a property, regression models were run to predict the price of the rental listing based on the basic attributes, such as location and number of rooms.  The features excluded from the model are ones that involves manager_id, or skill of management.  Only the listings that has “medium” interest are included in the training set for the price prediction modeling because the goal is to predict if a property is over- or under- valued, assuming that “medium” interest properties are fairly priced.    

Besides using a regression model to determine the predicted price, a simplified approach of using the median price of neighborhood with respect to the number of bedrooms are also used to determine the fair price.  The reasoning behind this approach is that this is how an average consumer gauge the price.   

The predicted price from both approaches are then used to create new features that aim to capture if a listing is over- or under-valued, and the magnitude of deviation from the fair price.  Some of these feaatures include the difference between predicted-price and listing-price, and the percent difference between the predicted and actual price. 

The other new features relating to price are price per bedroom, price per bathroom, and price per total number of rooms.

**Created** is a timestamp feature of when the listing was created.  Various combinations of year, month, day, weekday are extracted as there might be seasonality effect to the demand of the property.  The timing of listing also may have information, such as whether the listing is from an agency or an individual landlord.  For example, agencies are more likely to update the listing in the middle of the night in batches.  
