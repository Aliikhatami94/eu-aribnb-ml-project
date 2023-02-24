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

from data_processing import X_processed, y

# _________________________________________________________

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2, random_state=42, shuffle=True)

# Split the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)


# Get the scores
def get_scores(X_train, y_train, X_test, y_test, X_valid, y_valid, model_name, model):
    # Calculate the predictions
    fit_model(model_name, model)
    y_pred_train = model.predict(X_train)
    print('Training MAE:', mean_absolute_error(y_train, y_pred_train))
    print('Training RMSE:', np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print('Training R2:', r2_score(y_train, y_pred_train))

    y_pred_test = model.predict(X_test)
    print('\nTesting MAE:', mean_absolute_error(y_test, y_pred_test))
    print('Testing RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print('Testing R2:', r2_score(y_test, y_pred_test))

    y_pred_valid = model.predict(X_valid)
    print('\nValidation MAE:', mean_absolute_error(y_valid, y_pred_valid))
    print('Validation RMSE:', np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
    print('Validation R2:', r2_score(y_valid, y_pred_valid))


# Pick your model
def get_models(model_name):
    if model_name == 'xgb':
        return XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, early_stopping_rounds=5, random_state=42)
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


# Remove low importance features
def remove_low_features(X_train, X_test, X_valid, y_train, y_test, y_valid, model):
    # Find the feature importances
    perm = PermutationImportance(model, random_state=42).fit(X_valid, y_valid)

    # Show the feature importances in a dataframe
    weights_df = pd.DataFrame({'feature': X_valid.columns, 'weight': perm.feature_importances_})
    weights_df.sort_values('weight', ascending=False, inplace=True)
    weights_df.reset_index(drop=True, inplace=True)

    # Drop the features with a weight of less than 0.01 from the feature data set
    X_train.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)
    X_test.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)
    X_valid.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


# Set model
model_name = 'xgb'
model = get_models(model_name)
fit_model(model_name, model)

# Remove the low features
X_train, X_test, X_valid, y_train, y_test, y_valid = remove_low_features(X_train, X_test, X_valid, y_train, y_test, y_valid, model)

# Fit the model
model = get_models(model_name)
fit_model(model_name, model)

# Calculate the predictions
y_pred_train = model.predict(X_train)

# Get the scores
get_scores(X_train, y_train, X_test, y_test, X_valid, y_valid, model_name, model)