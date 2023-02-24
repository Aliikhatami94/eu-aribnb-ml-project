import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from data_transformation import X_processed
from data_processing import y

# _________________________________________________________

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2, random_state=42, shuffle=True)

# Split the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)


# Get the scores



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

