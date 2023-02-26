import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling import X_train, y_train, X_valid, y_valid, fit_model, model_name, model


# Get the accuracy scores
def get_scores(X_train, y_train, X_valid, y_valid, model_name, model):
    # Calculate the predictions
    fit_model(model_name, model)
    y_pred_train = model.predict(X_train)

    # Create a dictionary of the scores
    scores = {}

    # Calculate the scores
    scores['train_mae'] = mean_absolute_error(y_train, y_pred_train)
    scores['train_mse'] = mean_squared_error(y_train, y_pred_train)
    scores['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    scores['train_r2'] = r2_score(y_train, y_pred_train)

    # Calculate the predictions
    y_valid_pred = model.predict(X_valid)

    # Calculate the scores
    scores['valid_mae'] = mean_absolute_error(y_valid, y_valid_pred)
    scores['valid_mse'] = mean_squared_error(y_valid, y_valid_pred)
    scores['valid_rmse'] = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    scores['valid_r2'] = r2_score(y_valid, y_valid_pred)

    # Return the scores
    return scores


# Compare the predictions to the actual values
def compare_predictions(y_actual, y_pred):
    # Create a dataframe of the predictions and actual values
    predictions_df = pd.DataFrame({'actual': y_actual, 'predictions': y_pred})

    # Visualize actuals vs predictions with different colors for each
    plt.figure(figsize=(10, 10))
    plt.scatter(predictions_df.actual, predictions_df.predictions, c='blue', alpha=0.5, label='Predictions')
    plt.scatter(predictions_df.actual, predictions_df.actual, c='red', alpha=0.5, label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    plt.title('Actual vs Predictions')
    plt.legend(loc='best')
    plt.show()

    # Create a histogram of the residuals
    plt.figure(figsize=(10, 10))
    plt.hist(predictions_df.actual - predictions_df.predictions, bins=50)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Residuals Histogram')
    plt.show()

    # Convert the predictions back to the original scale
    predictions_df['predictions'] = np.exp(predictions_df['predictions'])
    predictions_df['actual'] = np.exp(predictions_df['actual'])

    print(predictions_df.head(10))


# Get the scores
score = get_scores(X_train, y_train, X_valid, y_valid, model_name, model)

print("Training predictions:")
compare_predictions(y_valid, model.predict(X_valid))

print("\nValidation predictions:")
compare_predictions(y_train, model.predict(X_train))

print('\nTraining R^2: ', score['train_r2'])
print('Validation R^2: ', score['valid_r2'])
