from modeling import *


# Remove low importance features
def remove_low_features(X_train, X_test, X_valid, y_train, y_test, y_valid, model):
    # Find the feature importances
    perm = PermutationImportance(model, random_state=42).fit(X_valid, y_valid)

    # Show the feature importances in a dataframe
    weights_df = pd.DataFrame({'feature': X_valid.columns, 'weight': perm.feature_importances_})
    weights_df.sort_values('weight', ascending=False, inplace=True)
    weights_df.reset_index(drop=True, inplace=True)

    # Drop the features with a weight of less than 0.01 from the feature data set
    X_train.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)
    X_test.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)
    X_valid.drop(weights_df[weights_df.weight < 0.01].feature, axis=1, inplace=True)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


# Remove the low features
X_train, X_test, X_valid, y_train, y_test, y_valid = remove_low_features(X_train, X_test, X_valid, y_train, y_test, y_valid, model)

# Fit the model
model = get_models(model_name)
fit_model(model_name, model)

# Calculate the predictions
y_pred_train = model.predict(X_train)

# Get the scores
get_scores(X_train, y_train, X_test, y_test, X_valid, y_valid, model_name, model)