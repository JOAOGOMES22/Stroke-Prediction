import os
import uuid
from logging import getLogger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize a logger for this module
logger = getLogger(__name__)

class StrokePredictionModel:
    """
    A class for building, training, and evaluating machine learning models for stroke prediction.
    Includes methods for model saving/loading and generating visualizations.

    Attributes:
        data (pd.DataFrame): Input dataset for model training.
        features (pd.DataFrame): Independent variables extracted from the dataset.
        labels (pd.Series): Target variable (Diagnosis) extracted from the dataset.
        model: Trained machine learning model.
    """
    def __init__(self, data=None):
        """
        Initialize the StrokePredictionModel with data and pre-process features and labels.

        Args:
            data (pd.DataFrame): Pre-processed dataset containing features and target variable.
        """
        self.data = data
        if data is not None:
            self.features = data.drop(columns=["Diagnosis"])  # Independent variables
            self.labels = data["Diagnosis"].astype(int)       # Target variable
        else:
            self.features = None
            self.labels = None
        self.model = None

    def train_model(self, model_type="RandomForest", params=None):
        """
        Train a machine learning model using the specified algorithm and hyperparameters.

        Args:
            model_type (str): Type of model to train. Options: 'RandomForest', 'SVM', 'GradientBoosting'.
            params (dict): Hyperparameter grid for fine-tuning using GridSearchCV.

        Returns:
            tuple: Test features (X_test) and test labels (y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        # Select model type
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == "SVM":
            self.model = SVC(probability=True, random_state=42)
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose 'RandomForest', 'SVM', or 'GradientBoosting'.")

        # Hyperparameter tuning
        if params:
            self.model = GridSearchCV(self.model, params, cv=5)

        # Train the model
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Log model performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        logger.info("\n" + classification_report(y_test, y_pred))

        return X_test, y_test
    
    def predict(self, new_data):
        """
        Make predictions using the trained model.

        Args:
            new_data (pd.DataFrame): Data for which predictions are required.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("No model loaded. Use 'load_model' to load a saved model.")
        return self.model.predict(new_data)
    
    def save_model(self, model_path="models/model.joblib"):
        """
        Save the trained model to the specified file path.

        Args:
            model_path (str): Path where the model will be saved.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            dump(self.model, model_path)
            logger.info(f"Model saved at: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path="models/model.joblib"):
        """
        Load a trained model from the specified file path.

        Args:
            model_path (str): Path to the saved model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        try:
            self.model = load(model_path)
            logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_prediction_graphs(self, X_test, y_test, static_folder):
        """
        Generate visualizations (confusion matrix and ROC curve) for the model predictions.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.
            static_folder (str): Directory to save the generated graphs.

        Returns:
            dict: Filenames of the generated graphs.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_filename = f"cm_{uuid.uuid4().hex}.png"
        cm_path = os.path.join(static_folder, cm_filename)
        plt.savefig(cm_path)
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_filename = f"roc_{uuid.uuid4().hex}.png"
        roc_path = os.path.join(static_folder, roc_filename)
        plt.savefig(roc_path)
        plt.close()

        logger.info(f"Generated graphs: {cm_path}, {roc_path}")
        return {"confusion_matrix": cm_filename, "roc_curve": roc_filename}
