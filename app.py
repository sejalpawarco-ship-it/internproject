# app.py

import sys

from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.spotify_song_explorer.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.spotify_song_explorer.components.data_transformation import DataTransformationConfig,DataTransformation
import mlflow 
import mlflow.sklearn   # अगर sklearn models log करने हैं



if __name__ == "__main__":
    logging.info("The execution has started")
    try:
        # Data ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data transformation
        data_transformation = DataTransformation()
        train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)                                                               

       #modelTrainer    

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
       
    
    except Exception as e:
        raise CustomException(e, sys)
        logging.info("CustomException")
     