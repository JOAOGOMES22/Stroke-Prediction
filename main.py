import logging
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify
from src.graphs_generate import GraphGenerator
from src.utils import setup_directories, load_csv, setup_logging
from src.data_processing import DataProcessor
from src.model_training import StrokePredictionModel

# Configuração do logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuração da aplicação Flask
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['STATIC_FOLDER'] = 'app/static'

# Variável global para armazenar os dados carregados
uploaded_data = None

# Rota para a página principal (upload)
@app.route('/')
def index():
    return render_template('index.html')

# Rota para processar o upload do arquivo CSV
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

        # Carregar os dados e armazenar na variável global
        uploaded_data = load_csv(file_path)
        logger.info("Arquivo CSV carregado com sucesso.")

        # Instanciar o gerador de gráficos
        graph_generator = GraphGenerator(uploaded_data, app.config['STATIC_FOLDER'])

        # Gerar todos os gráficos
        graphs = graph_generator.generate_all_graphs()

        # Renderizar a página com os gráficos gerados e a prévia dos dados
        return render_template(
            'index.html',
            data=uploaded_data.head().to_html(classes="table table-striped"),
            graphs=graphs,
            success=True
        )

    return "Por favor, envie um arquivo CSV válido."

# Rota para treinar o modelo
@app.route('/train', methods=['POST'])
def train_model():
    global uploaded_data

    if uploaded_data is None:
        logger.warning("Nenhum dado carregado para treinamento.")
        return "Nenhum dado carregado. Faça o upload de um arquivo CSV primeiro.", 400

    try:
        # Pré-processar os dados
        processor = DataProcessor(uploaded_data)
        processed_data = processor.process(target_column="Diagnosis")

        # Obter parâmetros do formulário
        model_type = request.form.get("model_type", "RandomForest")
        params = request.form.get("params")
        if params:
            params = eval(params)

        # Instanciar e treinar o modelo
        model = StrokePredictionModel(processed_data)
        X_test, y_test = model.train_model(model_type=model_type, params=params)
        
        prediction_graphs = model.generate_prediction_graphs(X_test, y_test, app.config['STATIC_FOLDER'])

        logger.info(f"Modelo {model_type} treinado com sucesso.")
        return render_template(
            "index.html",
            prediction_graphs=prediction_graphs,
            model_type=model_type
        )
    except Exception as e:
        logger.error(f"Erro ao treinar o modelo: {e}")
        return str(e), 500

# Rota para realizar predições
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber os dados para predição
        input_data = request.json
        if not input_data:
            return "Nenhum dado de entrada fornecido para predição.", 400

        input_df = pd.DataFrame([input_data])
        processor = DataProcessor(input_df)
        processed_data = processor.process()

        # Carregar o modelo treinado
        model = StrokePredictionModel(None)
        model.load_model()

        # Realizar a predição
        prediction = model.predict(processed_data)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return str(e), 500

# Configuração principal
if __name__ == '__main__':
    setup_directories(app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER'])
    app.run(debug=True)
