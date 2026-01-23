# app.py

import sys

from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.exception import CustomException

from src.spotify_song_explorer.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
)
from src.spotify_song_explorer.components.data_transformation import (
    DataTransformationConfig,
    DataTransformation,
)
from src.spotify_song_explorer.components.model_trainer import (
    ModelTrainerConfig,
    ModelTrainer,
)


if __name__ == "__main__":
    logging.info("The execution has started")
    try:
        # Data ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data transformation
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Model training
        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_array, test_array)
        print(f"R2 score: {r2:.4f}")

    except Exception as e:
        logging.info("CustomException")
        raise CustomException(e, sys)