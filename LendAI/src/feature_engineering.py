import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer new features for the LendAI model.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy so we don't modify the original dataframe
        X_copy = X.copy()
        
        # Total Income
        if 'ApplicantIncome' in X_copy.columns and 'CoapplicantIncome' in X_copy.columns:
            X_copy['TotalIncome'] = X_copy['ApplicantIncome'] + X_copy['CoapplicantIncome']
        
        # Loan Amount is usually in thousands, Total Income is raw. 
        # But we'll just calculate a raw ratio. Let's make sure we don't divide by zero.
        if 'LoanAmount' in X_copy.columns and 'TotalIncome' in X_copy.columns:
            # Add a small epsilon to avoid division by zero just in case
            X_copy['LoanToIncomeRatio'] = X_copy['LoanAmount'] / (X_copy['TotalIncome'] + 1e-5)
            
        # Log transformation for skewed features
        if 'LoanAmount' in X_copy.columns:
            # Filling any negative or zero with median or handling it. 
            # In our dataset LoanAmount should be > 0.
            X_copy['LoanAmountLog'] = np.log1p(X_copy['LoanAmount'])
            
        if 'TotalIncome' in X_copy.columns:
            X_copy['TotalIncomeLog'] = np.log1p(X_copy['TotalIncome'])
            
        return X_copy
