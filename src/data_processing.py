import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from logging import getLogger

logger = getLogger(__name__)

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataProcessor with the provided DataFrame.
        :param data: DataFrame to be processed.
        """
        self.data = data.copy()  # Creates a copy to avoid altering the original DataFrame

    def handle_missing_values(self):
        """
        Handles missing values in the DataFrame.
        - Numeric columns: Missing values are replaced with the column mean.
        - Categorical columns: Missing values are replaced with the column mode.
        """
        logger.info("Handling missing values...")
        
        # Handle missing values in numeric columns by replacing with the mean
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            if self.data[col].isnull().any():  # Check if there are missing values
                self.data[col].fillna(self.data[col].mean(), inplace=True)
                logger.info(f"Filled missing values with the mean in numeric column: {col}")

        # Handle missing values in categorical columns by replacing with the mode
        categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_columns:
            if self.data[col].isnull().any():  # Check if there are missing values
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                logger.info(f"Filled missing values with the mode in categorical column: {col}")

    def encode_categorical_variables(self):
        """
        Encodes categorical variables into numeric values using Label Encoding.
        """
        logger.info("Encoding categorical variables...")
        label_encoder = LabelEncoder()
        for column in self.data.select_dtypes(include=["object", "category"]).columns:
            # Transforms categorical data into numeric format
            self.data[column] = label_encoder.fit_transform(self.data[column])
            logger.info(f"Encoded categorical variable: {column}")

    def scale_features(self, target_column):
        """
        Normalizes numeric features to have mean 0 and standard deviation 1.
        :param target_column: The name of the target column to exclude from scaling.
        """
        logger.info("Normalizing numeric features...")
        scaler = StandardScaler()
        # Select numeric columns, excluding the target column
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        numeric_columns = numeric_columns.drop(target_column, errors="ignore")  # Exclude target column if it exists
        # Apply scaling to the selected numeric columns
        self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
        logger.info("Normalization completed for numeric features.")

    def process(self, target_column):
        """
        Executes the complete data preprocessing pipeline:
        - Handles missing values.
        - Encodes categorical variables.
        - Scales numeric features.
        :param target_column: The name of the target column to exclude from scaling.
        :return: The processed DataFrame.
        """
        logger.info("Starting data preprocessing pipeline...")
        self.handle_missing_values()  # Step 1: Handle missing values
        self.encode_categorical_variables()  # Step 2: Encode categorical variables
        self.scale_features(target_column)  # Step 3: Normalize numeric features
        logger.info("Data preprocessing completed.")
        return self.data  # Return the fully processed DataFrame
