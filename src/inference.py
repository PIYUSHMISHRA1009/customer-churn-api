import pandas as pd
import joblib
import os

# Paths
PROCESSED_DIR = "/Users/piyushkumarmishra/Downloads/mlops-customer-churn/data/processed"
MODEL_PATH = os.path.join(PROCESSED_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")

def load_model_and_preprocessor():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def predict(data: pd.DataFrame):
    model, preprocessor = load_model_and_preprocessor()

    # Preprocess the data
    X_transformed = preprocessor.transform(data)

    # Predict
    predictions = model.predict(X_transformed)
    prediction_probs = model.predict_proba(X_transformed)

    return predictions, prediction_probs

def main():
    # Example new customer data
    sample_data = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5
    }])

    predictions, probs = predict(sample_data)

    print("\n🎯 Prediction Result:")
    print(f"Churn: {'Yes' if predictions[0] == 1 else 'No'}")
    print(f"Probability: {probs[0][1]:.4f} (Churn), {probs[0][0]:.4f} (No Churn)")

if __name__ == "__main__":
    main()
