import numpy as np
from matplotlib import pyplot as plt

from data_processing import X, y


# Plot the distribution of the features
def plot_feature_dist(X):
    X.hist(figsize=(20, 20))
    plt.show()


# Log transform the data
def visualize_price_dist(log_y):
    # Visualize the distribution of the prices up to £1500
    plt.figure(figsize=(10, 5))
    log_y.hist(bins=100, range=(0, 1500))
    plt.margins(x=0)
    plt.axvline(log_y.mean(), color='orange', linestyle='--')
    plt.axvline(log_y.median(), color='red', linestyle='--')
    plt.title("Airbnb prices up to £1500", fontsize=16)
    plt.xlabel("Price (£)")
    plt.ylabel("Number of listings")
    plt.show()

    # Visualize the distribution of the prices from £1500 upwards
    plt.figure(figsize=(10, 5))
    log_y.hist(bins=100, range=(1500, max(log_y)))
    plt.margins(x=0)
    plt.axvline(log_y.mean(), color='orange', linestyle='--')
    plt.axvline(log_y.median(), color='red', linestyle='--')
    plt.title("Airbnb prices from £1500 upwards", fontsize=16)
    plt.xlabel("Price (£)")
    plt.ylabel("Number of listings")
    plt.show()

    print("Actual descriptive statistics:")
    print(y.describe())


# Log transform the data
def visualize_y_log_dsit(y):
    y = np.log(y)
    plt.figure(figsize=(10,5))
    y.hist(bins=100, range=(0, 10))
    plt.margins(x=0)

    # Change the x-axis to show the actual price rounded to the nearest £
    plt.xticks(np.arange(0, 10, step=1), np.exp(np.arange(0, 10, step=1)).round())

    plt.axvline(y.mean(), color='orange', linestyle='--')
    plt.axvline(y.median(), color='red', linestyle='--')
    plt.title("Log Transformed Airbnb prices", fontsize=16)
    plt.xlabel("Price (£)")
    plt.ylabel("Number of listings")
    plt.show()


plot_feature_dist(X)

visualize_y_log_dsit(y)
visualize_price_dist(y)

# Log transform the features
def visualize_X_log_dsit(X):
    num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    for col in num_cols:
        X[col] = X[col].astype('float64').replace(0.0, 0.01)
        X[col] = np.log(X[col])
    plt.figure(figsize=(10,5))
    X.hist(figsize=(10,11))
    plt.show()


visualize_X_log_dsit(X)