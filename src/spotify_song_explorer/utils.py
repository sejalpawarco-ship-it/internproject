import os
import sys  
from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql 

import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )    
        logging.info("Connection established")
        df = pd.read_sql_query("SELECT * FROM Spotify_song", mydb)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())
        return df
    
    except Exception as ex:
        raise CustomException(ex, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params[model_name]

            if model_name == "CatBoosting Regressor":
                # ⚠️ CatBoost does not support sklearn __sklearn_tags__, so skip GridSearchCV
                logging.info("Fitting CatBoost manually without GridSearchCV")
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                r2_square = r2_score(y_test, y_test_pred)
                report[model_name] = r2_square

            else:
                # Normal sklearn models can use GridSearchCV
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="r2")
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                best_model.fit(X_train, y_train)

                y_test_pred = best_model.predict(X_test)
                r2_square = r2_score(y_test, y_test_pred)
                report[model_name] = r2_square

        return report

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise CustomException(e, sys)
