import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_data_from_pickle

class PredictPipeline:
    def __init__(self):
        pass    
        # self.model = load_model(model_path)
    
    def predict(self, data:pd.DataFrame):
        try:
            # Prepare the input data for the model
            # input_data
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_data_from_pickle(model_path)

            # test_df = pd.read_csv("D:/Codebase2O/ml_project_structure/artifacts/test.csv").drop("math_score", axis=1)
            # test_df = test_df.iloc[1,:]
            # print('column are equal',test_df.columns == data.columns)
            # print(f'test columns name are {test_df.columns} \ndata column name are {data.columns}')
           
            preprocessor = load_data_from_pickle(preprocessor_path)
            data = preprocessor.transform(data)

            # Predict the math score 
            y = model.predict(data)

            return y
        except Exception as e:
            raise CustomException("Error while predicting data", e)

# gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
class CustomData:
    def __init__(self, 
                gender:str,
                race_ethnicity:str,
                parental_level_of_education:str,
                lunch:str,
                test_preparation_course:str,
                reading_score:int,
                writing_score:int
                ):
        self.gender = gender
        self.race_ethinicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_df(self):
        try:
            custom_data = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethinicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }

            return pd.DataFrame(custom_data)
        except Exception as e:
            raise CustomException("Error while converting data to dataframe", e)