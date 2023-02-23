import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_csv("Airbnb Prices in Europe/amsterdam_weekdays.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
