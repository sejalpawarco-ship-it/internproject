import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
import mlflow
import mlflow.sklearn

# 🔗 DagsHub integration
import dagshub
dagshub.init(
    repo_owner='sejalpawarco-ship-it',   # तुझ्या DagsHub repo owner
    repo_name='internproject',           # तुझ्या DagsHub repo name
    mlflow=True
)

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
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "XGB Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Best model selection
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Save best model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            # 🔥 MLflow logging (now connected to DagsHub)
            mlflow.set_experiment("spotify_song_explorer")

            with mlflow.start_run(run_name=f"{best_model_name}_run"):
                # Log parameters
                mlflow.log_param("model_name", best_model_name)
                mlflow.log_param("train_shape", X_train.shape)
                mlflow.log_param("test_shape", X_test.shape)

                # Log metrics
                mlflow.log_metric("best_model_score", best_model_score)
                mlflow.log_metric("r2_score", r2_square)

                # Save model in MLflow (DagsHub backend)
                mlflow.sklearn.log_model(best_model, "best_model")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)