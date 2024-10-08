import os
import sys
from dataclasses import dataclass #type: ignore

from catboost   import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_data_to_pickle
from src.utlis import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model(self,train_arry,test_arry):
        try:
            Xtrain, ytrain ,Xtest, ytest = (
                train_arry[:, :-1],
                train_arry[:, -1],
                test_arry[:, :-1],
                test_arry[:, -1],
            )


            # Define your models and their hyperparameter grids
            models = {
                "DecisionTree": (DecisionTreeRegressor(), {
                    'criterion':['absolute_error'],
                    'max_depth':[1,2,3,4,5],
                    # 'max_features':['auto'] , 
                    # 'min_samples_split':[2,5,10,],
                    # 'min_samples_leaf':[1,2,5,],
                    # 'min_weight_fraction_leaf':[0.1,0.2,0.3],
                    # 'max_leaf_nodes':[None,2,5,10],
                    # 'min_impurity_decrease':[0.0,0.1,0.2]
                }),
                "RandomForest": (RandomForestRegressor(), {
                    "max_depth": [5, 8, 15, None, 10],
                    "max_features": [5, 7, 8],
                    # "min_samples_split": [2, 8, 15, 20],
                    # "n_estimators": [50, 100, 150]}),
                }),
                "GradientBoosting": (GradientBoostingRegressor(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    # 'max_depth': [3, 5, 7]
                }),
                "AdaBoost": (AdaBoostRegressor(), {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }),
                "LinearRegression": (LinearRegression(), {}),
                "KNeighbors": (KNeighborsRegressor(), {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance']
                }),
                "DecisionTree": (DecisionTreeRegressor(), {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                "XGBoost": (XGBRegressor(), {
                    'n_estimators': [100, 200],
                    # 'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                }),
                "CatBoost": (CatBoostRegressor(silent=True), {
                    'iterations': [100, 200],
                    # 'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }),
            }

            metrics  = evaluate_models(
                models, Xtrain, ytrain, Xtest, ytest
            )

            best_model_score = max([model["test_r2"] for model in metrics.values()])
            best_model_name = [model for model in metrics.keys() if metrics[model]["test_r2"] == best_model_score][0]
            logging.info(f"Best model: {best_model_name}")

            best_model = metrics[best_model_name]["best_estimator"]
            

            save_data_to_pickle(best_model, self.model_trainer_config.trained_model_path)

            predicitons = best_model.predict(Xtest)
            r2_score_value = r2_score(ytest, predicitons)
            print(metrics)
            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)
        


# dict = {'DecisionTree': {'train_r2': 0.9349165733813759, 'test_r2': 0.7545258555984535, 'r2_difference': 0.18039071778292237, 'best_params': {'max_depth': 10, 'min_samples_split': 10}}, 'RandomForest': {'train_r2': 0.9017927339708544, 'test_r2': 0.8313892971467327, 'r2_difference': 0.07040343682412165, 'best_params': {'max_depth': 8, 'max_features': 8, 'min_samples_split': 15, 'n_estimators': 50}}, 'GradientBoosting': {'train_r2': 0.90801414768566, 'test_r2': 0.8522085401703967, 'r2_difference': 0.05580560751526331, 'best_params': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}}, 'AdaBoost': {'train_r2': 0.8323965940865241, 'test_r2': 0.7856847942671946, 'r2_difference': 0.04671179981932949, 'best_params': {'learning_rate': 0.1, 'n_estimators': 100}}, 'LinearRegression': {'train_r2': 0.8800829322886208, 'test_r2': 0.8605623436728859, 'r2_difference': 0.019520588615734913, 'best_params': {}}, 'KNeighbors': {'train_r2': 0.9996545814234725, 'test_r2': 0.7858732898247038, 'r2_difference': 0.21378129159876869, 'best_params': {'n_neighbors': 10, 'weights': 'distance'}}, 'XGBoost': {'train_r2': 0.9032391717369491, 'test_r2': 0.854093463021348, 'r2_difference': 0.04914570871560109, 'best_params': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}}, 'CatBoost': {'train_r2': 0.8977597959513891, 'test_r2': 0.8580313009240073, 'r2_difference': 0.03972849502738185, 'best_params': {'depth': 3, 'iterations': 200, 'learning_rate': 0.1}}}
# best_model_score = max([model["test_r2"] for model in dict.values()])
# best_model_name = [model for model in dict.keys() if dict[model]["test_r2"] == best_model_score][0]
# best_model_name
# print(max(best_model_score))
