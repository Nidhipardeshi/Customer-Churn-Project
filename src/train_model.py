import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/churn.csv")

# Select features
features = [
    'tenure','MonthlyCharges','TotalCharges',
    'Contract','PaymentMethod','Dependents',
    'InternetService','OnlineSecurity','TechSupport'
]

df = df[features + ['Churn']]

# Handle missing
df.dropna(inplace=True)

# Encode categorical
encoders = {}
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save model + encoders
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Model trained and saved")