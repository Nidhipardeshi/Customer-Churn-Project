import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
@st.cache_resource
def load_assets():
    model = joblib.load("models/churn_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, encoders

model, encoders = load_assets()

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# ---------------- INPUTS ---------------- #

tenure = st.slider("Tenure (Months)", 1, 72, 12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])
support = st.selectbox("Tech Support", ["Yes", "No"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict Churn"):

    try:
        # Create dataframe
        input_df = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            'Contract': [contract],
            'PaymentMethod': [payment],
            'Dependents': [dependents],
            'InternetService': [internet],
            'OnlineSecurity': [security],
            'TechSupport': [support]
        })

        # Apply encoding ONLY on categorical columns
        for col, encoder in encoders.items():
            if col in input_df.columns and input_df[col].dtype == 'object':
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    st.error(f"Invalid input for {col}")
                    st.stop()

        # Ensure correct column order
        input_df = input_df[model.feature_names_in_]

        # Prediction
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][prediction] * 100
        else:
            prob = 75.0

        # Output
        if prediction == 1:
            st.error(f"Customer likely to churn ({prob:.2f}%)")
        else:
            st.success(f"Customer likely to stay ({prob:.2f}%)")

    except Exception as e:
        st.error("Something went wrong. Check model and encoders.")
        st.exception(e)