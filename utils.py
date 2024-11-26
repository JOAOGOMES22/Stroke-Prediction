import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Força o backend "Agg"


def setup_directories(upload_folder, static_folder):
    """Cria os diretórios necessários para uploads e gráficos."""
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)


def load_csv(file_path) -> pd.DataFrame:
    """Carrega o arquivo CSV para um DataFrame do pandas."""
    return pd.read_csv(file_path)
