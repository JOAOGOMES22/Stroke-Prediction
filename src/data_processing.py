import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from logging import getLogger

logger = getLogger(__name__)

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa o processador de dados com o DataFrame fornecido.
        :param data: DataFrame a ser processado.
        """
        self.data = data.copy()

    def handle_missing_values(self):
        """Lida com valores ausentes de forma agnóstica."""
        logger.info("Lidando com valores ausentes...")
        
        # Preencher valores numéricos ausentes com a média
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mean(), inplace=True)
                logger.info(f"Valores ausentes preenchidos com a média na coluna numérica: {col}")

        # Preencher valores categóricos ausentes com a moda
        categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_columns:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                logger.info(f"Valores ausentes preenchidos com a moda na coluna categórica: {col}")

    def encode_categorical_variables(self):
        """Codifica variáveis categóricas para valores numéricos."""
        logger.info("Codificando variáveis categóricas...")
        label_encoder = LabelEncoder()
        for column in self.data.select_dtypes(include=["object", "category"]).columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])
            logger.info(f"Variável categórica codificada: {column}")

    def scale_features(self, target_column):
        """Normaliza as variáveis numéricas para média 0 e desvio padrão 1."""
        logger.info("Normalizando variáveis numéricas...")
        scaler = StandardScaler()
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        numeric_columns = numeric_columns.drop(target_column, errors="ignore")
        self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
        logger.info("Normalização concluída para variáveis numéricas.")

    def process(self, target_column):
        """Executa o pipeline completo de pré-processamento."""
        logger.info("Iniciando o pré-processamento dos dados...")
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.scale_features(target_column)
        logger.info("Pré-processamento concluído.")
        return self.data
    