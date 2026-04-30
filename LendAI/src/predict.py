import pandas as pd
from src.utils import load_config, load_object, get_logger

logger = get_logger('prediction_pipeline')

class Predictor:
    def __init__(self):
        self.config = load_config()
        try:
            self.preprocessor = load_object(self.config['model']['preprocessor_path'])
            self.model = load_object(self.config['model']['model_path'])
            logger.info("Model and preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            self.preprocessor = None
            self.model = None

    def predict(self, input_data: dict):
        """
        Takes raw dictionary input, preprocesses it, and returns prediction.
        """
        if not self.model or not self.preprocessor:
            return {"error": "Model not loaded"}

        try:
            # Convert single dictionary to DataFrame
            df = pd.DataFrame([input_data])
            
            # Preprocess
            X_processed = self.preprocessor.transform(df)
            
            # Predict
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0][1]
            
            return {
                "approved": bool(prediction == 1),
                "probability": float(probability)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

# For testing standalone
if __name__ == "__main__":
    test_input = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    predictor = Predictor()
    res = predictor.predict(test_input)
    print("Prediction:", res)
