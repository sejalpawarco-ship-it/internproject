# src/spotify_song_explorer/components/data_transformation.py

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def get_data_transformer_object(self):
        """Return a fitted/ready transformer object."""
        try:
            logging.info("Creating data transformer object")
            return StandardScaler()
        except Exception as e:
            raise CustomException(e, sys)

    def create_tempo_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create classification target variable - Tempo_bin."""
        try:
            df["Tempo_bin"] = pd.cut(
                df["Tempo"],
                bins=[0, 30, 60, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )
            logging.info("Tempo_bin created successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        try:
            # Encode Tempo_bin into a numeric column; name it tempo_bin_encoded for clarity
            df["tempo_bin_encoded"] = self.label_encoder.fit_transform(
                df["Tempo_bin"].astype(str)
            )
            logging.info("Categorical features encoded successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features_classification(self, df: pd.DataFrame):
        """Prepare features and target for classification."""
        try:
            feature_cols = [
                "Danceability",
                "Energy",
                "Year",
                "Tempo",
                "Popularity",
                "tempo_bin_encoded",
            ]
            X = df[feature_cols]
            y = df["Tempo_bin"]
            logging.info("Classification features prepared")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features_regression(self, df: pd.DataFrame):
        """Prepare features and target for regression."""
        try:
            feature_cols = [
                "Danceability",
                "Energy",
                "Tempo",
                "Year",
                "tempo_bin_encoded",
            ]
            X = df[feature_cols]
            y = df["Popularity"]
            logging.info("Regression features prepared")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def remove_missing_values(self, X: pd.DataFrame, y: pd.Series):
        """Remove rows with missing target values."""
        try:
            mask = ~pd.isnull(y).values
            X_clean = X.loc[mask]
            y_clean = y.loc[mask]
            removed_count = len(y) - len(y_clean)
            logging.info(f"Removed {removed_count} rows with missing values")
            return X_clean, y_clean
        except Exception as e:
            raise CustomException(e, sys)

    def split_and_scale_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """Split data into train-test and scale features."""
        try:
            X, y = self.remove_missing_values(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            logging.info(
                f"Data split and scaled - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
            )
            return X_train_scaled, X_test_scaled, y_train.values, y_test.values
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        End-to-end transformation for regression:
        - read train/test CSVs
        - create bins, encode categorical
        - prepare features/targets
        - fit scaler on train, transform both
        - save scaler and return numpy arrays for model trainer
        """
        try:
            logging.info("Reading training and testing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Creating Tempo_bin for classification")
            train_df = self.create_tempo_bins(train_df)
            test_df = self.create_tempo_bins(test_df)

            logging.info("Encoding categorical features")
            train_df = self.encode_categorical_features(train_df)
            test_df = self.encode_categorical_features(test_df)

            logging.info("Preparing features and target for regression")
            X_train, y_train = self.prepare_features_regression(train_df)
            X_test, y_test = self.prepare_features_regression(test_df)

            # Clean missing targets
            X_train, y_train = self.remove_missing_values(X_train, y_train)
            X_test, y_test = self.remove_missing_values(X_test, y_test)

            logging.info("Scaling train and test using train-fitted scaler")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Build final arrays (features + target as last column)
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Save the fitted scaler
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.scaler,
            )

            logging.info("Data transformation completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation_classification(self, train_path: str):
        """Complete pipeline for classification data transformation."""
        try:
            logging.info("Starting classification data transformation")
            df = pd.read_csv(train_path)
            logging.info("Data loaded for transformation")

            df = self.create_tempo_bins(df)
            df = self.encode_categorical_features(df)
            X, y = self.prepare_features_classification(df)

            X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.scaler,
            )

            logging.info("Classification data transformation completed")
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation_regression(self, train_path: str):
        """Complete pipeline for regression data transformation (single CSV)."""
        try:
            logging.info("Starting regression data transformation")
            df = pd.read_csv(train_path)
            logging.info("Data loaded for transformation")

            df = self.create_tempo_bins(df)
            df = self.encode_categorical_features(df)
            X, y = self.prepare_features_regression(df)

            X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.scaler,
            )

            logging.info("Regression data transformation completed")
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)