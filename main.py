import pandas as pd
import glob

data_list = glob.glob('Airbnb Prices in Europe/*.csv')

# Now we can use pandas to read all the csv files and combine them into one
df = pd.concat(map(pd.read_csv, data_list))
df.drop('Unnamed: 0', axis=1, inplace=True)
