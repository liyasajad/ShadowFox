import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.feature_engineering import FeatureEngineer
from src.utils import save_object

def get_preprocessor():
    """Builds and returns the scikit-learn preprocessing pipeline."""
    # Note: FeatureEngineer creates TotalIncome, LoanToIncomeRatio, LoanAmountLog, TotalIncomeLog
    numeric_features = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 
        'LoanToIncomeRatio', 'LoanAmountLog', 'TotalIncomeLog'
    ]
    
    categorical_features = [
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'Property_Area'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def split_data(df, target_col='Loan_Status', test_size=0.2, random_state=42):
    """Splits dataframe into train and test sets, formatting target variable."""
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    X = df.drop(target_col, axis=1)
    
    # Map target variable to binary
    y = df[target_col].map({'Y': 1, 'N': 0})
    if y.isnull().any():
        y = df[target_col]  # Fallback if already numeric
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
