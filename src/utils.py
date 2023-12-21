import os
import sys

import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        #print(e)
        raise CustomException(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.values())[i]] = test_model_score
        #print(report)
        return report
    except Exception as e:
        return CustomException(e,sys)