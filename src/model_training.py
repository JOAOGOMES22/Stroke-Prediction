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

logger = getLogger(__name__)

class StrokePredictionModel:
    def __init__(self, data=None):
        """
        Inicializa o modelo de predição de AVC.
        :param data: DataFrame pré-processado.
        """
        self.data = data
        if data is not None:
            self.features = data.drop(columns=["Diagnosis"])
            self.labels = data["Diagnosis"].astype(int)
        else:
            self.features = None
            self.labels = None
        self.model = None

    def train_model(self, model_type="RandomForest", params=None):
        """Treina um modelo de aprendizado de máquina."""
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        # Escolha do modelo
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == "SVM":
            self.model = SVC(probability=True, random_state=42)
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError("Modelo não suportado. Escolha 'RandomForest' ou 'SVM'.")

        # Ajuste de parâmetros
        if params:
            self.model = GridSearchCV(self.model, params, cv=5)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        logger.info("\n" + classification_report(y_test, y_pred))

        return X_test, y_test
    
    def predict(self, new_data):
        """Realiza predições usando o modelo carregado."""
        if self.model is None:
            raise ValueError("Modelo não carregado. Use 'load_model' para carregar o modelo salvo.")
        return self.model.predict(new_data)
    
    def save_model(self, model_path="models/model.joblib"):
        """Salva o modelo treinado no diretório especificado."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            dump(self.model, model_path)
            logger.info(f"Modelo salvo em: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar o modelo: {e}")
            raise

    def load_model(self, model_path="models/model.joblib"):
        """Carrega o modelo treinado do diretório especificado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"O arquivo do modelo não foi encontrado em: {model_path}")
        try:
            self.model = load(model_path)
            logger.info(f"Modelo carregado de: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            raise

    def generate_prediction_graphs(self, X_test, y_test, static_folder):
        """Gera gráficos relacionados às predições do modelo."""
        if self.model is None:
            raise ValueError("O modelo não foi treinado ou carregado.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusão")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        cm_filename = f"cm_{uuid.uuid4().hex}.png"
        cm_path = os.path.join(static_folder, cm_filename)
        plt.savefig(cm_path)
        plt.close()

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("Curva ROC")
        plt.xlabel("Falsos Positivos")
        plt.ylabel("Verdadeiros Positivos")
        plt.legend(loc="lower right")
        roc_filename = f"roc_{uuid.uuid4().hex}.png"
        roc_path = os.path.join(static_folder, roc_filename)
        plt.savefig(roc_path)
        plt.close()

        logger.info(f"Gráficos gerados: {cm_path}, {roc_path}")
        return {"confusion_matrix": cm_filename, "roc_curve": roc_filename}
