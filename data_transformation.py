from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from data_processing import X

# Get categorical columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

# Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Get numerical columns
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Log transform the numerical columns
for col in num_cols:
    X[col] = X[col].astype('float64').replace(0.0, 0.01)
    X[col] = np.log(X[col])

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

"""
# Plotting numerical features in a histogram to see their distribution of values
X[num_cols].hist(figsize=(20, 20))
plt.show()
"""