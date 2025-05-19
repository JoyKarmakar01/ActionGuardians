import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "actionguardian"

list_of_files = [
    ".github/workflows/.gitkeep",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_loader.py",
    f"src/{project_name}/components/preprocessor.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/predictor.py",
    f"src/{project_name}/components/summarizer.py",
    f"src/{project_name}/components/suggestor.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    f"src/{project_name}/constants/__init__.py",

    "config/config.yaml",
    "params.yaml",
    "schema.yaml",

    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",

    "research/eda.ipynb",
    "notebooks/demo_playground.ipynb",

    "templates/index.html",
    "static/style.css",

    "logs/running_logs.log",
    "tests/__init__.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"📁 Created directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"📄 Created empty file: {filepath}")
    else:
        logging.info(f"✅ File already exists: {filepath}")
