import logging
import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphGenerator:
    def __init__(self, data: pd.DataFrame, static_folder):
        self.data = data
        self.static_folder = static_folder

    def cleanup_graphs(self):
        """Remove gráficos antigos do diretório estático."""
        logger.info("Limpando gráficos antigos...")
        for file in os.listdir(self.static_folder):
            try:
                if file.startswith("graph_") and file.endswith(".png"):
                    os.remove(os.path.join(self.static_folder, file))
                    logger.info(f"Gráfico removido: {file}")
            except Exception as e:
                logger.error(f"Erro ao limpar gráfico {file}: {e}")

    def save_graph(self):
        """Salva o gráfico no diretório estático e retorna o nome do arquivo."""
        os.makedirs(self.static_folder, exist_ok=True)  # Garante que o diretório exista
        graph_filename = f'graph_{uuid.uuid4().hex}.png'
        graph_path = os.path.join(self.static_folder, graph_filename)

        try:
            plt.savefig(graph_path, dpi=300)  # Salva com alta qualidade
            plt.close()
            logger.info(f"Gráfico salvo em: {graph_path}")
            return graph_filename
        except Exception as e:
            logger.error(f"Erro ao salvar gráfico: {e}")
            return None

    def graph_age_distribution(self):
        """Gera o gráfico de distribuição de idades."""
        logger.info("Gerando gráfico de distribuição de idades.")
        self.data['Age Weight'] = self.data['Age'] / self.data['Age'].max()
        plt.figure(figsize=(8, 5))
        sns.histplot(
            x=self.data['Age'], bins=20, kde=True, weights=self.data['Age Weight'],
            alpha=0.5, label='All Patients'
        )
        sns.histplot(
            x=self.data[self.data['Stroke History'] == 1]['Age'], bins=20, kde=True,
            weights=self.data[self.data['Stroke History'] == 1]['Age Weight'],
            alpha=0.8, label='Stroke Patients'
        )
        plt.title('Age Distribution: All vs Stroke Patients', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.legend(title="Legend")
        plt.tight_layout()
        return self.save_graph()

    def graph_hypertension_diagnosis(self):
        """Gera o gráfico de hipertensão e diagnóstico."""
        logger.info("Gerando gráfico de hipertensão e diagnóstico.")
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Hypertension', hue='Diagnosis', data=self.data)
        plt.title('Hypertension and Stroke Diagnosis', fontsize=14)
        plt.xlabel('Hypertension (0=No, 1=Yes)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        return self.save_graph()

    def graph_glucose_levels(self):
        """Gera o gráfico de níveis de glicose."""
        plt.figure(figsize=(12, 8))
        sns.stripplot(
            x='Diagnosis',
            y='Average Glucose Level',
            data=self.data,
            jitter=True,
            size=8,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.6,
            palette="rocket"
        )
        plt.title('Impact of Glucose Levels on Stroke Diagnosis', fontsize=18, weight='bold')
        plt.xlabel('Stroke Diagnosis', fontsize=14)
        plt.ylabel('Average Glucose Level', fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        return self.save_graph()

    def graph_stress_levels_heatmap(self):
        """Gera o heatmap de níveis de estresse agrupados."""
        bins = [0, 2, 4, 6, 8, 10]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
        self.data['Stress Levels Grouped'] = pd.cut(self.data['Stress Levels'], bins=bins, labels=labels,
                                                    include_lowest=True)
        heatmap_data = self.data.pivot_table(
            index='Diagnosis',
            columns='Stress Levels Grouped',
            aggfunc='size',
            fill_value=0
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            cmap="rocket",
            fmt="d",
            cbar_kws={'label': 'Count'}
        )
        plt.title("Heatmap: Grouped Stress Levels by Stroke Diagnosis", fontsize=18, weight='bold')
        plt.xlabel("Grouped Stress Levels", fontsize=14)
        plt.ylabel("Stroke Diagnosis", fontsize=14)
        plt.tight_layout()
        return self.save_graph()

    def graph_alcohol_intake(self):
        """Gera o gráfico de ingestão de álcool."""
        plt.figure(figsize=(12, 8))
        sns.histplot(
            data=self.data,
            x="Alcohol Intake",
            hue="Stroke History",
            multiple="dodge",
            palette="rocket",
            shrink=0.8,
            kde=True,
            edgecolor="black"
        )
        plt.title("Alcohol Intake vs Stroke History", fontsize=18, weight='bold')
        plt.xlabel("Alcohol Intake", fontsize=14)
        plt.ylabel("Count/Density", fontsize=14)
        plt.tight_layout()
        return self.save_graph()

    def graph_physical_activity_heatmap(self):
        """Gera o heatmap de atividade física e histórico de AVC."""
        heatmap_data_balanced = self.data.groupby(['Physical Activity', 'Stroke History']).size().unstack(fill_value=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data_balanced,
            cmap="rocket",
            annot=True,
            fmt="d",
            linewidths=0.5,
            cbar_kws={'label': 'Number of Patients'}
        )
        plt.title("Physical Activity vs Stroke History: Heatmap Analysis (Balanced)", fontsize=18, weight='bold')
        plt.xlabel("Stroke History", fontsize=14)
        plt.ylabel("Physical Activity", fontsize=14)
        plt.tight_layout()
        return self.save_graph()

    def generate_all_graphs(self):
        """Gera todos os gráficos e retorna a lista de arquivos gerados."""
        logger.info("Iniciando a geração de todos os gráficos.")
        self.cleanup_graphs()
        graphs = [self.graph_age_distribution(), self.graph_hypertension_diagnosis(), self.graph_glucose_levels(),
                  self.graph_stress_levels_heatmap(), self.graph_alcohol_intake(),
                  self.graph_physical_activity_heatmap()]
        return graphs
