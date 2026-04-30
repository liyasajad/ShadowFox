# LendAI

LendAI is a production-ready Machine Learning system designed to predict loan approvals based on structured financial data. It features a complete ML workflow including data preprocessing, feature engineering, model training, evaluation, and a prediction API.

## Project Structure

```
LendAI/
│
├── data/
│   ├── raw/
│   │   └── loan_prediction.csv
│   └── processed/
│
├── models/                  # Pickled models and preprocessors are saved here
│
├── src/
│   ├── data_preprocessing.py # Scikit-learn pipeline for imputation, scaling, and encoding
│   ├── feature_engineering.py # Custom features (TotalIncome, LoanToIncomeRatio, Log transforms)
│   ├── train.py              # Script to train, tune, and save the Random Forest model
│   ├── evaluate.py           # Script to test the model and display metrics/feature importance
│   ├── predict.py            # Predictor class for standalone inference
│   └── utils.py              # Common utilities (logging, config parsing)
│
├── api/
│   └── app.py                # Flask API exposing a /predict endpoint
│
├── config/
│   └── config.yaml           # Central configuration for paths and model hyperparameters
│
└── requirements.txt          # Project dependencies
```

## Setup Instructions

1. **Install Python 3.8+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Model Training

To train the Random Forest model and generate the preprocessor pipeline, run:

```bash
python src/train.py
```

This will output the tuned Random Forest metrics and save `lendai_model.pkl` and `preprocessor.pkl` in the `models/` folder.

## Model Evaluation

To evaluate the saved model on the test dataset and view the confusion matrix, ROC-AUC, and feature importances, run:

```bash
python src/evaluate.py
```

## Running the API

Start the Flask server to expose the prediction endpoint:

```bash
python api/app.py
```

The server will run on `http://127.0.0.1:5000`.

## API Usage Example

Send a `POST` request to the `/predict` endpoint:

**Using cURL:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
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
}'
```

**Response format:**
```json
{
    "approved": true,
    "probability": 0.84
}
```
