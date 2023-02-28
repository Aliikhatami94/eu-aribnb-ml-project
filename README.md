# Airbnb Prices in Europe Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
2. [Libraries used](#libraries-used)
3. [Visualizations](#visualizations)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model](#model)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

## Introduction
This code performs analysis of Airbnb prices in Europe. It reads all the csv files in the folder and combines them into one using pandas. The data is then preprocessed by dropping unnecessary columns, checking for null values, and cross-featuring the longitude and latitude.

## Data
Link to the dataset: https://www.sciencedirect.com/science/article/pii/S0261517721000388?via%3Dihub#sec3

|price | log of the final sum |
| --- |----------------------|
|bedrooms |	number of bedrooms |
|person_capacity |	maximum number of guests |
|room_type |	type of room |
|room_private |	dummy for private rooms |
|room_shared |	dummy for shared rooms |
|cleanliness |	guest reviews: scale to 10 |
|guest_satisfaction |	guest reviews: scale to 100 |
|superhost |	dummy for hosts with the superhost status |
|multi |	dummy for listings offered by hosts with 2–4 listings |
|biz |	dummy for listings offered by hosts with more than 4 listings |
|dist |	distance to the city centre in kilometres |
|metro_dist |	distance to the closest metro station in kilometres |
|attr_index |	attraction index: scale to 100 |
|rest_index |	restaurant index: scale to 100 |


## Libraries used
The following libraries are used in this analysis:

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* xgboost
* time


## Visualizations
Various visualizations are used to better understand the data, including:

* A histogram of the distribution of prices
* A bar plot of the distribution of the satisfaction rating
* Heatmap of correlation between features
* Scatter plot of predicted vs. actual values
* Histogram of residuals

![Collinearity_heatmap.png](img%2FCollinearity_heatmap.png)


## Data Preprocessing
* Dropping unnecessary columns
* Binning categorical columns
* Scaling numerical features
* Replacing 0 values with 0.01 to avoid errors in log transformation


## Feature Engineering
* Cross-featuring the longitude and latitude
* Log transforming the numerical columns to improve normality of distributions

## Models

### Random Forest Regression:
* Random forests regressors are an ensemble learning method for regression that operate by constructing a multitude of decision trees at training time and outputting the mean prediction of the individual trees.
* The random forest regressor was used to compare the performance of the XGBoost model.
* The random forest regressor was not used in the final model because it performed worse than the XGBoost model.
* The random forest regressor was able to achieve an R-squared value of 75% for the validation data and 78% for the training data.

```
Training MSE: 0.075
Validation MSE: 0.0838

Training r2: 0.776
Validation r2: 0.7485

MSE training-validation gap: 0.0088
r2 training-validation gap: 0.0274
```

### XGBoost Regression:
* XGBoost is an advanced implementation of gradient boosting which optimizes the gradient boosting algorithm using a number of techniques such as parallel processing, tree pruning, regularization and handling missing values. XGBoost works by iteratively training a sequence of decision trees, where each tree attempts to correct the errors of the previous tree.
* Feature importances are visualized to determine which features are most important in predicting the price of an Airbnb listing.
* The XGBoost model was able to achieve an R-squared value of 76% for the validation data and 79% for the training data.
* The XGBoost model was used in the final model because it performed better than the random forest regressor.

```
Training MSE: 0.071
Validation MSE: 0.0798

Training r2: 0.7881
Validation r2: 0.7607

Difference between training and validation scores:
MSE: 0.0088
r2: 0.0273
```

![Feature importances.png](img%2FFeature%20importances.png)


## Conclusion
This analysis provides insights into the factors that contribute to the price of an Airbnb listing in Europe. It shows the importance of features such as the number of bedrooms, the distance from the city center, and the satisfaction rating. The XGBoost regression model was able to accurately predict the price of a listing with an R-squared value of 76% for the validation data and 79% for the training data.


![Actuals_vs_Predicted.png](img%2FActuals_vs_Predicted.png)

* Most resiuals are centered around 0, indicating that the model is performing well.

![Residuals.png](img%2FResiduals.png)

### Future Work

1. Include additional features - There are other factors that could influence the price of an Airbnb listing that were not considered in this analysis, such as the amenities available, the number of reviews, and the time of year. Adding these features could improve the accuracy of the model.

2. Experiment with different models - Although the XGBoost model worked well in this analysis, there are many other models that could be tested, such as Random Forest, Support Vector Machines, and Neural Networks. Experimenting with different models could lead to improved performance.

3. Consider other regions - This analysis focused on Airbnb prices in Europe, but it could be interesting to extend it to other regions such as North America, Asia, or South America.

4. Use more recent data - The dataset used in this analysis has a knowledge cutoff of September 2021. Using more recent data could provide more accurate insights into the current state of Airbnb prices in Europe.

5. Perform cluster analysis - It could be interesting to group the Airbnb listings into clusters based on their features and use a separate model to predict the prices for each cluster.

6. Include external data sources - Incorporating external data sources such as data on local events, transportation options, or crime rates could provide additional insights into the factors that influence Airbnb prices.

7. Perform time series analysis - It could be interesting to perform time series analysis on the Airbnb prices to see how they have changed over time and their trends.

8. Perform sentiment analysis - It could be interesting to perform sentiment analysis on the reviews to determine if there is a correlation between the sentiment of the reviews and the price of the listing.

9. Perform topic modeling - It could be interesting to perform topic modeling on the reviews to determine if there are certain topics that are more common in reviews for expensive listings.



