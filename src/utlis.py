import pickle
import os
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def save_data_to_pickle(data, file_path):
    """
    Save the given data to a pickle file.

    Parameters:
    data (any): The data to be saved.
    filename (str): The name of the file where data will be saved.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        dill.dump(data, file)

# Example usage:
# data = {'key': 'value'}
# save_data_to_pickle(data, 'data.pkl')
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluate multiple models and return their metrics on training and test data.

    Parameters:
    models (dict): A dictionary where keys are model names and values are model instances and their hyperparameter grids.
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Test data features.
    y_test (array-like): Test data labels.

    Returns:
    dict: A dictionary containing the metrics for each model.
    """
    metrics = {}
    # best_model_name = None
    # best_r2_score = -np.inf
    # best_r2_difference = np.inf
    # best_model = None

    for name, (model, param_grid) in models.items():
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_

        # Predict on training and test data
        y_train_pred = best_estimator.predict(X_train)
        y_test_pred = best_estimator.predict(X_test)

        # Calculate R^2 scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        r2_difference = abs(train_r2 - test_r2)

        # Store metrics
        metrics[name] = {
            'best_estimator': best_estimator,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_difference': r2_difference,
            'best_params': grid_search.best_params_
        }



    return metrics