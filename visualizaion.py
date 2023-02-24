from data_processing import X, y

from matplotlib import pyplot as plt

# Plotting numerical features in a histogram to see their distribution of values
X.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()