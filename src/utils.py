import os
import pandas as pd
import matplotlib
import logging
import os

matplotlib.use('Agg')


def setup_directories(upload_folder, static_folder):
    """Cria os diretórios necessários para uploads e gráficos."""
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)


def load_csv(file_path) -> pd.DataFrame:
    """Carrega o arquivo CSV para um DataFrame do pandas."""
    return pd.read_csv(file_path)


def setup_logging():
    logger = logging.getLogger()
    if not logger.hasHandlers():  # Evita adicionar múltiplos handlers
        logger.setLevel(logging.INFO)

        # Formato do log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
