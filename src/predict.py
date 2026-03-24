import joblib
import pandas as pd

model = joblib.load('models/churn_model.pkl')

# Example input
sample = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 70,
    "TotalCharges": 800
}])

prediction = model.predict(sample)

print("Prediction:", prediction)