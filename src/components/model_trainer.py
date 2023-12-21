import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")

            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "random forest":RandomForestRegressor(),
                "decisiontree": DecisionTreeRegressor(),
                "gradient boosting":GradientBoostingRegressor(),
                "linear regression": LinearRegression(),
                "knn regressor": KNeighborsRegressor(),
                "xgbregressor":XGBRegressor(),
                "adaboost classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            #print(model_report)
            ## get the best model score

            best_model_score = max(sorted(model_report.values()))

            ## get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            #best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model_name
            )
            logging.info("the best model is saved : {}".format(best_model_name))

            #predicted = models[best_model_name].predict(x_test)
            r2_squared = best_model_score

            return r2_squared





        except Exception as e:
            #print(e)
            return CustomException(e,sys)



