import os
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

# Project name
project_name = "spotify_song_explorer"

# List of files and folders to create
list_of_files = [
    ".github/workflows/.gitkeep",

    f"src/{project_name}/__init__.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/feature_engineering.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",

    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",

    "main.py"
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "README.md"
]

def create_project_structure():
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir = filepath.parent

        # Create directories if they don't exist
        if not filedir.exists():
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory: {filedir}")

        # Create empty file if it doesn't exist
        if not filepath.exists():
            with open(filepath, "w") as f:
                pass
            logging.info(f"Created file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")

if __name__ == "__main__":
    create_project_structure()
    logging.info("✅ Project structure created successfully!")
