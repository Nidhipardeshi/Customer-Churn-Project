# Customer Churn Prediction

## Overview
This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to leave and help businesses take proactive retention actions.

---
## Features
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Advanced Data Visualization
- Feature Engineering
- Machine Learning Models (Logistic Regression, Random Forest)
- Model Evaluation (Accuracy, Confusion Matrix)
- Feature Importance Analysis

---
## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib

---
## Project Structure
Customer-Churn-Project/
│
├── data/ # Dataset
├── notebooks/ # Jupyter notebooks (EDA & analysis)
├── src/ # Source code (model training)
├── models/ # Saved ML model
├── requirements.txt # Dependencies


---
## Key Insights
- Customers with **month-to-month contracts** have higher churn rates  
- Higher **monthly charges** are linked with increased churn  
- Customers with **longer tenure** are less likely to churn  
- Certain services and payment methods influence customer retention  

---
## Model Performance
- Logistic Regression (Baseline Model)
- Random Forest (Best Performing Model)
- Evaluated using accuracy and classification metrics

---
## Installation

Clone the repository:
git clone https://github.com/Nidhipardeshi/Customer-Churn-Project.git

Install dependencies:

pip install -r requirements.txt
How to Run

Run the model training script:
python src/train_model.py

## Results
Achieved good prediction accuracy using Random Forest
Identified key factors affecting customer churn
Provided actionable insights for customer retention

## Future Improvements
Hyperparameter tuning for better accuracy
Model deployment using Streamlit or Flask
Real-time prediction system
