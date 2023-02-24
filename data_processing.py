import pandas as pd
import glob
import numpy as np

data_list = glob.glob('Airbnb Prices in Europe/*.csv')

# Now we can use pandas to read all the csv files and combine them into one
df = pd.concat(map(pd.read_csv, data_list))
df.drop('Unnamed: 0', axis=1, inplace=True)

# Cross feature longitude and latitude
df['lat_lng'] = df['lat'] * df['lng']

# Drop the latitude and longitude columns
df.drop(['lat', 'lng'], axis=1, inplace=True)

# Set X and y
X = df.drop(['realSum', 'attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm'], axis=1)
y = df.realSum

# Log transform the data
y = np.log(y)

