import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.datatransformation_config=DataTransformationConfig()

    def get_datatransformer_obj(self):
        """
        This function is responsible for data transformation based on 
        different types of data"""

        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())

                ]
            )

            logging.info(f"Numerical column encoding completed: {numerical_features}")

            cat_pipeline =  Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical column encoding completed {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ("categorical_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("read train/test data completed")

                logging.info("opening preprocessing object")

                preprocessor = self.get_datatransformer_obj()

                target_col_name = "math_score"
                numerical_features = ['writing_score','reading_score']

                input_features_train_df = train_df.drop(columns=[target_col_name],axis=1)
                target_features_train_df = train_df[target_col_name]

                input_features_test_df = test_df.drop(columns=[target_col_name],axis=1)
                target_features_test_df = test_df[target_col_name]

                logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

                input_feature_train = preprocessor.fit_transform(input_features_train_df)
                input_feature_test = preprocessor.transform(input_features_test_df)

                train_arr = np.c_[input_feature_train,np.array(target_features_train_df)]
                test_arr = np.c_[input_feature_test,np.array(target_features_test_df)]

                logging.info(f"Saved preprocessing object.")

                save_object(

                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor

            )
                return (
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path,
            )

            except Exception as e:
                raise CustomException(e,sys)
