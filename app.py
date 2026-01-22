from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.components.data_ingestion import DataIngestion
from src.spotify_song_explorer.components.data_ingestion import DataIngestionConfig
from src.spotify_song_explorer.components.data_transformation import DataTransformationConfig,DataTransformation
import sys


if __name__ == "__main__":
    logging.info("The execution has started")

    try: 
      # data_ingestion_config=DataIngestionConfig()
      data_ingestion=DataIngestion()
      train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

      # data_ingestion.initiate_data_ingestion()
      data_transformation=DataTransformation()
      data_transformation.initiate_data_transformation()
      
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
