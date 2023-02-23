import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_csv("Airbnb Prices in Europe/amsterdam_weekdays.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Find categorical columns and numerical columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('realSum')

# Define the transformers for numerical and categorical columns
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle the transformers
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

# Apply the transformations to the data
df = preprocessor.fit_transform(df)

df = pd.DataFrame(df)
