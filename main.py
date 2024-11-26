import logging
from flask import Flask, render_template, request, redirect, url_for
from graphs_generate import GraphGenerator  # Classe para gerar gráficos
from utils import setup_directories, load_csv  # Funções auxiliares
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuração da aplicação Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['STATIC_FOLDER'] = 'static'
app.template_folder = 'app/templates'


# Rota para a página principal (upload)
@app.route('/')
def index():
    return render_template('upload.html')


# Rota para processar o upload do arquivo CSV
@app.route('/upload', methods=['POST'])
def upload_file():
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

        # Processar os dados
        data = load_csv(file_path)
        logger.info("Arquivo CSV carregado com sucesso.")

        # Instanciar o gerador de gráficos
        graph_generator = GraphGenerator(data, app.config['STATIC_FOLDER'])

        # Gerar todos os gráficos
        graphs = graph_generator.generate_all_graphs()

        # Renderizar a página com os gráficos gerados e a prévia dos dados
        return render_template(
            'upload.html',
            data=data.head().to_html(classes="table table-striped"),
            graphs=graphs,
            success=True
        )

    return "Por favor, envie um arquivo CSV válido."


# Configuração principal
if __name__ == '__main__':
    setup_directories(app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER'])
    app.run(debug=True)
