import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Força o backend "Agg"
import matplotlib.pyplot as plt
import uuid


def setup_directories(upload_folder, static_folder):
    """Cria os diretórios necessários para uploads e gráficos."""
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)


def load_csv(file_path):
    """Carrega o arquivo CSV para um DataFrame do pandas."""
    return pd.read_csv(file_path)


def process_data(data):
    """Processa os dados aplicando uma transformação numérica."""
    processed_data = {}
    for col in data.columns:
        try:
            numeric_values = pd.to_numeric(data[col], errors='coerce')  # Converter para numérico
            processed_data[col] = np.tanh(numeric_values.fillna(0).values)  # Aplicar transformação
        except Exception as e:
            print(f"Erro ao processar a coluna {col}: {e}")
    return processed_data


def generate_graphs(data, static_folder):
    """Gera gráficos a partir dos dados processados e os salva como arquivos PNG."""
    # Limpa gráficos antigos
    cleanup_graphs(static_folder)

    graphs = []
    for i, (col, values) in enumerate(data.items(), start=1):
        if i > 6:  # Limitar a 6 gráficos
            break
        plt.figure()
        plt.plot(values, label=col, color=np.random.rand(3, ))
        plt.title(f'Graph {i}')
        plt.legend()
        graph_filename = f'graph_{uuid.uuid4().hex}.png'
        graph_path = os.path.join(static_folder, graph_filename)
        plt.savefig(graph_path)
        plt.close()
        graphs.append(graph_filename)  # Apenas o nome do arquivo
    return graphs


def cleanup_graphs(static_folder):
    """Remove gráficos antigos do diretório estático."""
    for file in os.listdir(static_folder):
        if file.startswith("graph_") and file.endswith(".png"):
            os.remove(os.path.join(static_folder, file))
