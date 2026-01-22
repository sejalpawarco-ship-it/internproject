import os
import sys

from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from src.spotify_song_explorer.utils import save_object

from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def initiate_data_transformation(self):
        self.transformation_config=DataTransformationConfig()
        self.label_encoder=LabelEncoder()
        self.scaler=StandardScaler()

    def create_tempo_bins(self, df):
        """Create classification target variable - Tempo_bin"""
        try:
            df["Tempo_bin"] = pd.cut(
                df["Tempo"], 
                bins=[0, 30, 60, 100], 
                labels=["Low", "Medium", "High"]
            )
            logging.info("Tempo_bin created successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def encode_categorical_features(self, df):
        """Encode categorical features using LabelEncoder"""
        try:
            df["genre_encoded"] = self.label_encoder.fit_transform(df["Tempo_bin"].astype(str))
            logging.info("Categorical features encoded successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features_classification(self, df):
        """Prepare features and target for classification"""
        try:
            X = df[["Danceability", "Energy", "Year", "Tempo", "Popularity", "genre_encoded"]]
            y = df["Tempo_bin"]
            logging.info("Classification features prepared")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features_regression(self, df):
        """Prepare features and target for regression"""
        try:
            X = df[["Danceability", "Energy", "Tempo", "Year", "genre_encoded"]]
            y = df["Popularity"]
            logging.info("Regression features prepared")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def remove_missing_values(self, X, y):
        """Remove rows with missing target values"""
        try:
            mask = ~pd.isnull(y).values
            X_clean = X[mask]
            y_clean = y[mask]
            removed_count = len(y) - len(y_clean)
            logging.info(f"Removed {removed_count} rows with missing values")
            return X_clean, y_clean
        except Exception as e:
            raise CustomException(e, sys)

    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train-test and scale features"""
        try:
            X, y = self.remove_missing_values(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logging.info(f"Data split and scaled - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation_classification(self, train_path):
        """Complete pipeline for classification data transformation"""
        try:
            logging.info("Starting classification data transformation")
            
            df = pd.read_csv(train_path)
            logging.info("Data loaded for transformation")
            
            df = self.create_tempo_bins(df)
            df = self.encode_categorical_features(df)
            X, y = self.prepare_features_classification(df)
            
            X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
            
            logging.info("Classification data transformation completed")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation_regression(self, train_path):
        """Complete pipeline for regression data transformation"""
        try:
            logging.info("Starting regression data transformation")
            
            df = pd.read_csv(train_path)
            logging.info("Data loaded for transformation")
            
            df = self.create_tempo_bins(df)
            df = self.encode_categorical_features(df)
            X, y = self.prepare_features_regression(df)
            
            X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
            
            logging.info("Regression data transformation completed")
            return X_train, X_test, y_train, y_test
        
            save_object(
                file_path=self.data_transdformation_config.preprocessor_obj_file_path,
                obj=self.scaler
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
 