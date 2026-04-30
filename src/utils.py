import yaml
import logging
import pickle
import os

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_logger(name):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_object(obj, filepath):
    """Saves a python object to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_object(filepath):
    """Loads a python object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
