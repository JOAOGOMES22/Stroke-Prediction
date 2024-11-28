import os
import pandas as pd
import matplotlib
import logging

# Configure matplotlib to use the 'Agg' backend for non-GUI environments
matplotlib.use('Agg')


def setup_directories(upload_folder, static_folder):
    """
    Create necessary directories for file uploads and static content.

    Args:
        upload_folder (str): Path to the folder where uploaded files will be stored.
        static_folder (str): Path to the folder where static files (e.g., graphs) will be saved.
    """
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)


def load_csv(file_path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path)


def setup_logging():
    """
    Configure global logging settings.

    - Sets the log level to INFO.
    - Formats log messages with timestamps and log levels.
    - Ensures only a single handler is attached to prevent duplicate logs.

    Adds:
        - Console output for logs.
    """
    logger = logging.getLogger()
    if not logger.hasHandlers():  # Avoid adding multiple handlers to the logger
        logger.setLevel(logging.INFO)

        # Define the log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
