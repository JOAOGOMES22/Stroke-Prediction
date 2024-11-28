import logging
from src.utils import setup_logging
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging settings for the application
setup_logging()
logger = logging.getLogger(__name__)  # Creates a logger instance for this module


class GraphGenerator:
    """
    A utility class for generating and managing graphs based on a dataset.

    Attributes:
        data (pd.DataFrame): The dataset used for generating graphs.
        static_folder (str): Directory where generated graphs are stored.
    """
    def __init__(self, data: pd.DataFrame, static_folder):
        """
        Initialize the GraphGenerator with dataset and output directory.

        Args:
            data (pd.DataFrame): Input data for visualizations.
            static_folder (str): Path to the directory for saving generated graphs.
        """
        self.data = data
        self.static_folder = static_folder

    def cleanup_graphs(self):
        """
        Remove previously generated graphs from the static directory.

        This ensures that old graphs are deleted before new ones are generated, 
        avoiding clutter in the output directory.
        """
        logger.info("Cleaning up old graphs...")
        for file in os.listdir(self.static_folder):
            try:
                if file.startswith("graph_") and file.endswith(".png"):
                    os.remove(os.path.join(self.static_folder, file))
                    logger.info(f"Removed old graph: {file}")
            except Exception as e:
                logger.error(f"Error while cleaning graph {file}: {e}")

    def save_graph(self):
        """
        Save the current matplotlib figure as a PNG file in the static directory.

        Returns:
            str: The filename of the saved graph if successful, otherwise None.
        """
        os.makedirs(self.static_folder, exist_ok=True)
        graph_filename = f'graph_{uuid.uuid4().hex}.png'
        graph_path = os.path.join(self.static_folder, graph_filename)

        try:
            plt.savefig(graph_path, dpi=300)  # Saves the plot with high resolution
            plt.close()
            logger.info(f"Graph saved at: {graph_path}")
            return graph_filename
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return None

    def graph_age_distribution(self):
        """
        Generate and save a histogram showing the age distribution of patients.

        The histogram distinguishes between patients with and without stroke history.
        """
        logger.info("Generating age distribution graph.")
        self.data['Age Weight'] = self.data['Age'] / self.data['Age'].max()
        plt.figure(figsize=(8, 5))
        sns.histplot(
            x=self.data['Age'], bins=20, kde=True, weights=self.data['Age Weight'],
            alpha=0.5, label='All Patients'
        )
        sns.histplot(
            x=self.data[self.data['Stroke History'] == 1]['Age'], 
            bins=20, 
            kde=True,
            weights=self.data[self.data['Stroke History'] == 1]['Age Weight'],
            alpha=0.8, 
            label='Stroke Patients', 
            palette='rocket',
        )
        plt.title('Age Distribution: All vs Stroke Patients', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.legend(title="Legend")
        plt.tight_layout()
        return self.save_graph()

    def graph_hypertension_diagnosis(self):
        """
        Generate and save a bar chart comparing hypertension presence and stroke diagnosis.
        """
        logger.info("Generating hypertension vs diagnosis graph.")
        plt.figure(figsize=(8, 5))
        sns.countplot(
            x='Hypertension', 
            hue='Diagnosis', 
            data=self.data,
            palette='rocket',
        )
        plt.title('Hypertension and Stroke Diagnosis', fontsize=14)
        plt.xlabel('Hypertension (0=No, 1=Yes)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        return self.save_graph()

    def graph_glucose_levels(self):
        """
        Generate and save a strip plot showing glucose levels by stroke diagnosis.
        """
        logger.info("Generating glucose levels graph.")
        plt.figure(figsize=(12, 8))
        sns.stripplot(
            x='Diagnosis',
            y='Average Glucose Level',
            data=self.data,
            hue='Diagnosis',
            jitter=True,
            size=8,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.6,
            palette="rocket",
        )
        plt.title('Impact of Glucose Levels on Stroke Diagnosis', fontsize=18, weight='bold')
        plt.xlabel('Stroke Diagnosis', fontsize=14)
        plt.ylabel('Average Glucose Level', fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        return self.save_graph()

    def graph_stress_levels_heatmap(self):
        """
        Generate and save a heatmap showing stress levels grouped by stroke diagnosis.
        """
        logger.info("Generating stress levels heatmap.")
        bins = [0, 2, 4, 6, 8, 10]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
        self.data['Stress Levels Grouped'] = pd.cut(
            self.data['Stress Levels'], bins=bins, labels=labels, include_lowest=True
        )
        heatmap_data = self.data.pivot_table(
            index='Diagnosis',
            columns='Stress Levels Grouped',
            aggfunc='size',
            fill_value=0,
            observed=False,
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
        """
        Generate and save a histogram comparing alcohol intake with stroke history.
        """
        logger.info("Generating alcohol intake vs stroke history graph.")
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
        """
        Generate and save a heatmap showing physical activity levels by stroke history.
        """
        logger.info("Generating physical activity vs stroke history heatmap.")
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
        """
        Generate all graphs and return a list of filenames for the saved graphs.

        This method orchestrates the generation of multiple graphs for analysis.
        """
        logger.info("Starting the generation of all graphs.")
        self.cleanup_graphs()
        graphs = [
            self.graph_age_distribution(),
            self.graph_hypertension_diagnosis(),
            self.graph_glucose_levels(),
            self.graph_stress_levels_heatmap(),
            self.graph_alcohol_intake(),
            self.graph_physical_activity_heatmap()
        ]
        return graphs
