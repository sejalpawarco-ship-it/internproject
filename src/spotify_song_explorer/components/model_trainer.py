# src/spotify_song_explorer/components/model_trainer.py

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
from src.spotify_song_explorer.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "spotify_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(
                    random_state=42, objective="reg:squarederror", n_estimators=200
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False, random_state=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            params = {
                "Random Forest": {"n_estimators": [64, 128, 256]},
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7, 9]},
                "XGB Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [100, 200, 300],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [100, 200, 300],
                },
                "AdaBoost Regressor": {"n_estimators": [64, 128, 256]},
            }

            # Evaluate all models (assumes evaluate_models returns dict of best scores and fits models)
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            # Best model score and name
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with score {best_model_score}"
            )

            # Fit best model on full training data before saving/predicting
            best_model.fit(X_train, y_train)

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)