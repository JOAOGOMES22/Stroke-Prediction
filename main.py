import logging
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify
from src.data_analysis import GraphGenerator
from src.utils import setup_directories, load_csv, setup_logging
from src.data_processing import DataProcessor
from src.model_training import StrokePredictionModel
from src.config import SELECTED_COLUMNS

# Configuração do logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuração da aplicação Flask
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['STATIC_FOLDER'] = 'app/static'
app.config['MODEL_PATH'] = 'models/model.joblib'

# Variável global para armazenar os dados carregados
uploaded_data = None

# Rota para a página principal (upload)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data

    if 'file' not in request.files:
        logger.warning("Nenhum arquivo foi enviado no formulário.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        logger.warning("Arquivo enviado está vazio.")
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info(f"Arquivo salvo em: {file_path}")

        # Carregar e pré-processar os dados
        raw_data = load_csv(file_path)
        logger.info("Arquivo CSV carregado com sucesso. Iniciando o pré-processamento...")
        processor = DataProcessor(raw_data[SELECTED_COLUMNS + ["Diagnosis"]])
        uploaded_data = processor.process(target_column="Diagnosis")
        logger.info("Dados pré-processados com sucesso.")

        # Gerar gráficos com os dados originais (opcional, pode usar uploaded_data também)
        graph_generator = GraphGenerator(raw_data, app.config['STATIC_FOLDER'])
        graphs = graph_generator.generate_all_graphs()

        # Renderizar a página com os gráficos gerados e a prévia dos dados
        return render_template(
            'index.html',
            data=raw_data.head().to_html(classes="table table-striped"),
            graphs=graphs,
            success=True
        )

    return "Por favor, envie um arquivo CSV válido."


@app.route('/train', methods=['POST'])
def train_model():
    global uploaded_data

    if uploaded_data is None:
        logger.warning("Nenhum dado carregado para treinamento.")
        return "Nenhum dado carregado. Faça o upload de um arquivo CSV primeiro.", 400

    try:
        # Obter parâmetros do formulário
        model_type = request.form.get("model_type", "RandomForest")
        params = request.form.get("params")
        if params:
            params = eval(params)

        # Instanciar e treinar o modelo com os dados já preprocessados
        model = StrokePredictionModel(uploaded_data[SELECTED_COLUMNS + ["Diagnosis"]])
        X_test, y_test = model.train_model(model_type=model_type, params=params)
        
        # Salvar o modelo
        model.save_model(app.config['MODEL_PATH'])
        logger.info(f"Modelo {model_type} treinado e salvo com sucesso.")

        # Gerar gráficos de predição
        prediction_graphs = model.generate_prediction_graphs(X_test, y_test, app.config['STATIC_FOLDER'])

        return render_template(
            "index.html",
            prediction_graphs=prediction_graphs,
            model_type=model_type,
        )
    except Exception as e:
        logger.error(f"Erro ao treinar o modelo: {e}")
        return str(e), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        if not input_data:
            return "Nenhum dado de entrada fornecido para predição.", 400
        
        # Converter valores para os tipos apropriados
        for key, value in input_data.items():
            try:
                if value.isdigit():
                    input_data[key] = int(value)  # Converter para int se for um número inteiro
                else:
                    input_data[key] = float(value) if "." in value else value
            except ValueError:
                pass

        input_df = pd.DataFrame([input_data])
        input_df = input_df[SELECTED_COLUMNS]
        if input_df.empty:
            return jsonify({"error": "Dados de entrada inválidos."}), 400

        # Pré-processar os dados de entrada
        processor = DataProcessor(input_df)
        processed_data = processor.process(target_column=None)

        # Carregar o modelo treinado
        model = StrokePredictionModel(None)
        model.load_model(app.config['MODEL_PATH'])

        # Realizar a predição
        prediction = model.predict(processed_data)
        result = "Alta probabilidade de AVC" if prediction[0] == 1 else "Baixa probabilidade de AVC"
        return render_template("index.html", prediction_result=result)
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return str(e), 500

# Configuração principal
if __name__ == '__main__':
    setup_directories(app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER'])
    app.run(debug=True)
