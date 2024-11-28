import logging
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify
from src.data_analysis import GraphGenerator
from src.utils import setup_directories, load_csv, setup_logging
from src.data_processing import DataProcessor
from src.model_training import StrokePredictionModel
from src.config import SELECTED_COLUMNS

# Logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Flask application configuration
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.config['UPLOAD_FOLDER'] = 'app/uploads'  # Directory for uploaded files
app.config['STATIC_FOLDER'] = 'app/static'  # Directory for static content (e.g., graphs)
app.config['MODEL_PATH'] = 'models/model.joblib'  # Path for saving the trained model

# Global variable to store uploaded and processed data
uploaded_data = None

# Route for the home page
@app.route('/')
def index():
    """
    Render the home page, allowing users to upload CSV files.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads, process the data, and generate graphs.

    - Checks if a valid CSV file is uploaded.
    - Loads and preprocesses the data.
    - Generates exploratory data analysis graphs.
    """
    global uploaded_data

    if 'file' not in request.files:
        logger.warning("No file provided in the request.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        logger.warning("Uploaded file is empty.")
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info(f"File saved at: {file_path}")

        # Load and preprocess the data
        raw_data = load_csv(file_path)
        logger.info("CSV file successfully loaded. Starting preprocessing...")
        processor = DataProcessor(raw_data[SELECTED_COLUMNS + ["Diagnosis"]])
        uploaded_data = processor.process(target_column="Diagnosis")
        logger.info("Data preprocessing completed successfully.")

        # Generate graphs (optional: use raw_data or processed data)
        graph_generator = GraphGenerator(raw_data, app.config['STATIC_FOLDER'])
        graphs = graph_generator.generate_all_graphs()

        # Render the home page with graphs and a data preview
        return render_template(
            'index.html',
            data=raw_data.head().to_html(classes="table table-striped"),
            graphs=graphs,
            success=True
        )

    return "Please upload a valid CSV file."

@app.route('/train', methods=['POST'])
def train_model():
    """
    Train a machine learning model using the uploaded data.

    - Retrieves parameters and model type from the form.
    - Trains the model and saves it to disk.
    - Generates prediction-related graphs.
    """
    global uploaded_data

    if uploaded_data is None:
        logger.warning("No data uploaded for training.")
        return "No data uploaded. Please upload a CSV file first.", 400

    try:
        # Get parameters from the form
        model_type = request.form.get("model_type", "RandomForest")
        params = request.form.get("params")
        if params:
            params = eval(params)  # Evaluate string as Python expression

        # Train the model
        model = StrokePredictionModel(uploaded_data[SELECTED_COLUMNS + ["Diagnosis"]])
        X_test, y_test = model.train_model(model_type=model_type, params=params)
        
        # Save the trained model
        model.save_model(app.config['MODEL_PATH'])
        logger.info(f"Model {model_type} trained and saved successfully.")

        # Generate prediction-related graphs
        prediction_graphs = model.generate_prediction_graphs(X_test, y_test, app.config['STATIC_FOLDER'])

        return render_template(
            "index.html",
            prediction_graphs=prediction_graphs,
            model_type=model_type,
        )
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions based on user input.

    - Processes user-provided input.
    - Loads the trained model and predicts the probability of stroke.
    """
    try:
        input_data = request.form.to_dict()
        if not input_data:
            return "No input data provided for prediction.", 400
        
        # Convert input values to appropriate types
        for key, value in input_data.items():
            try:
                if value.isdigit():
                    input_data[key] = int(value)  # Convert to int if input is a whole number
                else:
                    input_data[key] = float(value) if "." in value else value
            except ValueError:
                pass

        input_df = pd.DataFrame([input_data])
        input_df = input_df[SELECTED_COLUMNS]
        if input_df.empty:
            return jsonify({"error": "Invalid input data."}), 400

        # Preprocess the input data
        processor = DataProcessor(input_df)
        processed_data = processor.process(target_column=None)

        # Load the trained model
        model = StrokePredictionModel(None)
        model.load_model(app.config['MODEL_PATH'])

        # Make prediction
        prediction = model.predict(processed_data)
        result = "High stroke probability" if prediction[0] == 1 else "Low stroke probability"
        return render_template("index.html", prediction_result=result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return str(e), 500

# Main application configuration
if __name__ == '__main__':
    # Ensure necessary directories exist before starting the application
    setup_directories(app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER'])
    app.run(debug=True)
