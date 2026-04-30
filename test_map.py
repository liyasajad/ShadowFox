import pandas as pd
df = pd.read_csv("data/raw/loan_prediction.csv")
y = df['Loan_Status'].map({'Y': 1, 'N': 0})
print("Mapped:", y.head())
print("Unique:", y.unique())
print("dtype:", y.dtype)
