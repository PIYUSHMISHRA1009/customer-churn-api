from fastapi import FastAPI
from src.inference_api.schema import CustomerData
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Load model and preprocessor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # project root
model_path = os.path.join(BASE_DIR, "data", "processed", "model.joblib")
preprocessor_path = os.path.join(BASE_DIR, "data", "processed", "preprocessor.joblib")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)[0]
    proba = model.predict_proba(transformed_input)[0]

    result = {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": {
            "Churn": round(float(proba[1]), 4),
            "No Churn": round(float(proba[0]), 4)
        }
    }

    return result
