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
import eli5
from eli5.sklearn import PermutationImportance


data_list = glob.glob('Airbnb Prices in Europe/*.csv')

# Now we can use pandas to read all the csv files and combine them into one
df = pd.concat(map(pd.read_csv, data_list))
df.drop('Unnamed: 0', axis=1, inplace=True)

# Set X and y
X = df.drop('realSum', axis=1)
y = df.realSum

# Get numerical and categorical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

# Preprocessing for numerical data
num_transformer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use ColumnTransformer to combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Use the pipeline to process the data
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaler', scaler)])
X_processed = pipeline.fit_transform(X)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Split the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# Define the model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1)

# Fit the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

perm = PermutationImportance(model, random_state=42).fit(X_valid, y_valid)
weights = eli5.format_as_dataframe(eli5.explain_weights(perm))

# Drop features with weights of less than 0.03
X_train = X_train[:, weights.weight > 0.01]
X_test = X_test[:, weights.weight > 0.01]
X_valid = X_valid[:, weights.weight > 0.01]

# Define the model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1)

# Fit the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Get a evaluation function that returns all the accruacy metrics
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('R2: ', r2)
    return mae, mse, rmse, r2

# Evaluate the model on the training set
print('Training set')
evaluate_model(model, X_train, y_train)

# Evaluate the model on the testing set
print('\nTesting set')
evaluate_model(model, X_test, y_test)

# Evaluate the model on the validation set
print('\nValidation set')
evaluate_model(model, X_valid, y_valid)