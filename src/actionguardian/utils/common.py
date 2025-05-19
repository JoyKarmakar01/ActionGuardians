import os
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any
from src.actionguardian import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Load a YAML configuration file and return it as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the file is empty.
        Exception: For any other I/O or parsing errors.

    Returns:
        ConfigBox: Parsed YAML content.
    """
    try:
        with open(path_to_yaml, 'r') as f:
            content = yaml.safe_load(f)
            if content is None:
                raise BoxValueError("Empty YAML")
            logger.info(f"YAML loaded: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        logger.error(f"YAML file is empty: {path_to_yaml}")
        raise ValueError(f"YAML file {path_to_yaml} is empty.")
    except Exception as e:
        logger.error(f"Failed to load YAML {path_to_yaml}: {e}")
        raise

@ensure_annotations
def save_yaml(path: Path, data: dict):
    """
    Save a dictionary to a YAML file.

    Args:
        path (Path): Destination YAML file path.
        data (dict): Data to serialize.
    """
    with open(path, 'w') as f:
        yaml.dump(data, f)
    logger.info(f"YAML saved: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a JSON file.

    Args:
        path (Path): Destination JSON file path.
        data (dict): Data to serialize.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file and return it as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Parsed JSON content.
    """
    with open(path, 'r') as f:
        content = json.load(f)
    logger.info(f"JSON loaded: {path}")
    return ConfigBox(content)

@ensure_annotations
def create_directories(paths: list, verbose: bool = True):
    """
    Create multiple directories, if they don't already exist.

    Args:
        paths (list): List of directory paths (str or Path).
        verbose (bool): Whether to log directory creation.
    """
    for p in paths:
        os.makedirs(p, exist_ok=True)
        if verbose:
            logger.info(f"Directory created or exists: {p}")

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Serialize an object to disk with joblib.

    Args:
        data (Any): Object to serialize.
        path (Path): Path to the output .pkl file.
    """
    joblib.dump(data, path)
    logger.info(f"Binary saved: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a joblib‐serialized object from disk.

    Args:
        path (Path): Path to the .pkl file.

    Returns:
        Any: The deserialized Python object.
    """
    obj = joblib.load(path)
    logger.info(f"Binary loaded: {path}")
    return obj
