import os
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # type: ignore


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self ):
        self.ingesion_config = DataIngestionConfig()
    
    # read data  from databases
    def initiate_data_ingestion(self):

        try:
            logging.info('Data Ingestion Initiated')
            # notebook\data\stud.csv
            data = pd.read_csv('notebook/data/stud.csv')

            os.makedirs(os.path.dirname(self.ingesion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingesion_config.raw_data_path, index=False,header=True)


            logging.info("Train test split initiated")
            train_data, test_data = train_test_split(data, test_size=0.2)

            train_data.to_csv(self.ingesion_config.train_data_path, index=False,header=True)
            test_data.to_csv(self.ingesion_config.test_data_path, index=False,header=True)

            logging.info('Data Ingestion Completed')

            return data
        except Exception as e:
            # print('hello')
            raise CustomException(e, sys)
        
        return (self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
                )
    
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

