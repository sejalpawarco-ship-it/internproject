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
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )

class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        self.transformation_config=DataTransformationConfig()
        self.label_encoder=LabelEncoder()
        self.scaler=StandardScaler()
        try:
            logging.info("Reading training and testing data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Creating Tempo_bin for classification")
            
            train_df=self.create_tempo_bins(train_df)
            test_df=self.create_tempo_bins(test_df)
            
            logging.info("Encoding categorical features")
            train_df=self.encode_categorical_features(train_df)
            test_df=self.encode_categorical_features(test_df)
            
            logging.info("Preparing features and target for regression")
            X_train, y_train=self.prepare_features_regression(train_df)
            X_test, y_test=self.prepare_features_regression(test_df)
            
            logging.info("Splitting and scaling data")
            X_train_scaled, y_train = self.split_and_scale_data(X_train, y_train)
            X_test_scaled, y_test = self.split_and_scale_data(X_test, y_test)

            logging.info("Scaling train and test data")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            logging.info("Data transformation completed")
            return train_arr, test_arr
            
        except Exception as e:
            raise CustomException(e, sys)

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
 