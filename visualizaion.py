import numpy as np
from matplotlib import pyplot as plt

from data_processing import X, y


# Plot the distribution of the features
def plot_feature_dist(X):
    X.hist(figsize=(20, 20))
    plt.show()


# Log transform the data
def visualize_actual_dist(log_y):
    plt.figure(figsize=(20,4))
    log_y.hist(bins=100, range=(0, 2000))
    plt.margins(x=0)
    plt.axvline(log_y.mean(), color='orange', linestyle='--')
    plt.axvline(log_y.median(), color='red', linestyle='--')
    plt.title("Airbnb prices", fontsize=16)
    plt.xlabel("Price (£)")
    plt.ylabel("Number of listings")
    plt.show()

    print("Actual descriptive statistics:")
    print(y.describe())


# Log transform the data
def visualize_log_dsit(y):
    y = np.log(y)
    plt.figure(figsize=(20,4))
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

    print(f"Log transformed")
    print("____________________")
    print(f"Mean: {np.exp(y.mean())}")
    print(f"Median: {np.exp(y.median())}")


plot_feature_dist(X)
visualize_actual_dist(y)
print("\n")
visualize_log_dsit(y)