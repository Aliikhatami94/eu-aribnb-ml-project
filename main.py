import pandas as pd
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Read the data
df = pd.read_csv("Airbnb Prices in Europe/amsterdam_weekdays.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Split the data into features and target
X = df.drop(['realSum'], axis=1)
y = df['realSum']

# Find categorical columns and numerical columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define the transformers for numerical and categorical columns
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle the transformers
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

# Remove outliers form the entire dataset
z_scores = stats.zscore(np.array(df[num_cols]))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]

# _________________________________________________________________

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Apply the transformations to the data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


# Create the model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

# Fit the model
xgb.fit(X_train_processed, y_train)

# Predict the prices
y_pred_train = xgb.predict(X_train_processed)
y_pred_val = xgb.predict(X_val_processed)
y_pred_test = xgb.predict(X_test_processed)



def print_performance(y_true, y_pred):
    """
    This function takes the true values and predicted values and prints the MAE, RMSE, and R-squared metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'R-squared: {r2:.2f}')


# Evaluate the model
print("Training set performance:")
print_performance(y_train, y_pred_train)
print('\nValidation set performance:')
print_performance(y_val, y_pred_val)
print('\nTest set performance:')
print_performance(y_test, y_pred_test)