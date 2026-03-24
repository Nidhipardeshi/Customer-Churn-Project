import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Sample input (change values to test)
input_data = {
    'tenure': 12,
    'MonthlyCharges': 70,
    'TotalCharges': 840,
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check',
    'Dependents': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'TechSupport': 'No'
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Apply encoding
for col in encoders:
    df[col] = encoders[col].transform(df[col])

# Prediction
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][prediction] * 100

# Output
print("\n===== Prediction Result =====")

if prediction == 1:
    print(f"Customer will churn ❌ ({probability:.2f}%)")
else:
    print(f"Customer will stay ✅ ({probability:.2f}%)")