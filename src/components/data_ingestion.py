import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                
            )
        except Exception as e:
            return CustomException(e,sys)
        
if __name__ == "__main__":
    #Step 1 : Data ingestion here training and test data will be saved a s.csv file
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion()


    #Step 2 : Data transformation step. Here numerical and categorical variables are 
    # transformed and are divided into train set and test set  with the data from step 1
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)


    # Step 3: Model trainer. Here different models are trained on the train and test set from step 2 and they are evaluated based on R2 
    # score and the best model is saved as .pkl file.
    #Also returns the r2 score for the best model

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_training(train_arr,test_arr))






