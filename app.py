from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.components.data_ingestion import DataIngestion
from src.spotify_song_explorer.components.data_ingestion import DataIngestionConfig
from src.spotify_song_explorer.components.data_transformation import DataTransformationConfig,DataTransformation
from src.spotify_song_explorer.components.model_trainer import ModelTrainer,ModelTrainerConfig
import sys


if __name__ == "__main__":
    logging.info("The execution has started")

    try: 
      # data_ingestion_config=DataIngestionConfig()
      data_ingestion=DataIngestion()
      train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

      # data_transformation_config=DataTransformationConfig()
      data_transformation=DataTransformation()
      train_arr,test_arr=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

      ## Model Training
      model_trainer=ModelTrainer()
      print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
