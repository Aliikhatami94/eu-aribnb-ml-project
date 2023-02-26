from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

from data_transformation import X_processed
from data_processing import y

# _________________________________________________________

# Based on the log transformed data, Setting any price above 1500 to £1500
y = np.where(y > 1500, 1500, y)

# Setting any price below £50 to £50
y = np.where(y < 50, 50, y)

# Log transform the target variable
y = np.log(y)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2, random_state=42, shuffle=True)

# Split the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Pick your model
def get_models(model_name):
    if model_name == 'xgb':
        return XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, early_stopping_rounds=5, random_state=42, reg_lambda=0.1)
    elif model_name == 'rf':
        return RandomForestRegressor(n_estimators=500, random_state=42)
    elif model_name == 'lr':
        return LinearRegression()


# Fit the model
def fit_model(model_name, model):
    if model_name == 'xgb':
        return model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_name == 'rf':
        return model.fit(X_train, y_train)
    elif model_name == 'lr':
        return model.fit(X_train, y_train)


# Set model
model_name = 'xgb'
model = get_models(model_name)
fit_model(model_name, model)

# Calculate the predictions
y_pred_train = model.predict(X_train)

