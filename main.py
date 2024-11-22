from flask import Flask, render_template, request, redirect, url_for
from utils import setup_directories, load_csv, process_data, generate_graphs  # Importa as funções auxiliares
import os

# Configuração da aplicação Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['STATIC_FOLDER'] = 'app/static'
app.template_folder = 'app/templates'


# Rota para a página principal (upload)
@app.route('/')
def index():
    return render_template('upload.html')


# Rota para processar o upload do arquivo CSV
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Processar os dados
        data = load_csv(file_path)
        processed_data = process_data(data)

        # Gerar gráficos com base nos dados processados
        graphs = generate_graphs(processed_data, app.config['STATIC_FOLDER'])

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
