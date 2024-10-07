# from src.components.data_ingestion import DataIngestion
import sys
import os
from dataclasses import dataclass # type: ignore


import pandas as pd
import numpy as np
from  sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_data_to_pickle
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()     
    
    def get_data_transformer_object(self):

        '''This function is used to get the data transformation object'''	
        try:
            # numerical_columns = ['math_score', 'reading_score', 'writing_score']
            numerical_columns = [ 'reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    # ('std_scaler', StandardScaler())
                ] 
            )

            logging.info(f'Data Transformation Initiated with \n numerical columns: {numerical_columns} and \ncategorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ] ,
                n_jobs=-1
            )

            return preprocessor
        except Exception as e:

            raise CustomException(e, sys)
        
    def initiate_preprocessor(self,train_path,test_path):
        try:
            logging.info('Data Transformation Initiated')
            preprocessor_obj = self.get_data_transformer_object()
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            logging.info('Data Transformation Completed')

            target_column_name = "math_score"
            
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(f"\n\n\n{input_feature_train_df.columns}")

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]

            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            file_path = self.transformation_config.preprocessor_obj_file_path

            save_data_to_pickle(preprocessor_obj, file_path)

            logging.info('Saved preprocessor object')
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
                                        )
        except Exception as e:
            raise CustomException(e, sys)