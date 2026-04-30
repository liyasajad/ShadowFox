import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from src.predict import Predictor
from src.utils import get_logger

app = Flask(__name__)
logger = get_logger('api_app')

# Initialize predictor
try:
    predictor = Predictor()
    logger.info("Predictor initialized successfully in API.")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": predictor is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None or predictor.model is None:
        return jsonify({"error": "Model is not loaded or unavailable."}), 503
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400
            
        logger.info(f"Received prediction request: {data}")
        
        # We can add more specific validation here if needed
        required_fields = ['ApplicantIncome', 'LoanAmount', 'Credit_History']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
            
        # Get prediction
        result = predictor.predict(data)
        
        if "error" in result:
            return jsonify(result), 500
            
        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return jsonify({"error": "Internal server error during prediction."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
