import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data_list = glob.glob('Airbnb Prices in Europe/*.csv')

# Now we can use pandas to read all the csv files and combine them into one
df = pd.concat(map(pd.read_csv, data_list))
df.drop('Unnamed: 0', axis=1, inplace=True)

# Set X and y
X = df.drop('realSum', axis=1)
y = df.realSum

# Get categorical columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

# Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Get numerical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
num_transformer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Use ColumnTransformer to combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Get the preprocessed data back into a dataframe with the column names
X_processed = pd.DataFrame(preprocessor.fit_transform(X))
X_processed.columns = num_cols + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))