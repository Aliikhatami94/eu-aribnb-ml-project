# Airbnb Prices in Europe Analysis

This code performs analysis of Airbnb prices in Europe. It reads all the csv files in the folder and combines them into one using pandas. The data is then preprocessed by dropping unnecessary columns, checking for null values, and cross-featuring the longitude and latitude.

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

![Feature importances.png](img%2FFeature%20importances.png)


## Model
* XGBoost regression is used to predict the price of an Airbnb listing
* Feature importances are visualized to determine which features are most important in predicting the price of an Airbnb listing


## Conclusion
This analysis provides insights into the factors that contribute to the price of an Airbnb listing in Europe. It shows the importance of features such as the number of bedrooms, the distance from the city center, and the satisfaction rating. The XGBoost regression model was able to accurately predict the price of a listing with an R-squared value of 76% for the validation data and 82% for the training data.

```
Training MSE: 0.0625
Validation MSE: 0.0802
Training r2: 0.8192
Validation r2: 0.7623
```

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



