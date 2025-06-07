from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # <-- Added CORS
from src.inference_api.schema import CustomerData
import joblib
import pandas as pd
import os
import logging
import traceback
from time import time
from contextlib import asynccontextmanager

# -------------------------------
# Logging Configuration
# -------------------------------
LOG_FILE = "logs/api.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------
# Load Model and Preprocessor on Import (startup)
# -----------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/inference_api

    model_path = os.path.join(BASE_DIR, "model", "model.joblib")
    preprocessor_path = os.path.join(BASE_DIR, "model", "preprocessor.joblib")

    logging.info(f"🔍 Looking for model at: {model_path}")
    logging.info(f"🔍 Looking for preprocessor at: {preprocessor_path}")
    logging.info(f"📁 Model exists? {os.path.exists(model_path)}")
    logging.info(f"📁 Preprocessor exists? {os.path.exists(preprocessor_path)}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    logging.info("✅ Model and Preprocessor loaded successfully.")

except Exception as e:
    logging.error(f"❌ Failed to load model or preprocessor: {e}")
    raise RuntimeError("Model or preprocessor could not be loaded. Check paths!")

# -------------------------------
# Lifespan event handler (startup/shutdown)
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 API is starting up...")
    yield
    logging.info("🛑 API is shutting down...")

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Enable CORS for Frontend Access
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
def read_root():
    logging.info("📡 Health check called.")
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict")
async def predict_churn(data: CustomerData, request: Request):
    start = time()
    try:
        logging.info(f"📥 Received input: {data.model_dump()}")
        input_df = pd.DataFrame([data.model_dump()])
        transformed_input = preprocessor.transform(input_df)

        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0]

        result = {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": {
                "Churn": round(float(probability[1]), 4),
                "No Churn": round(float(probability[0]), 4)
            }
        }
        duration = time() - start
        logging.info(f"✅ Prediction successful in {duration:.3f} seconds: {result}")
        return result

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed. Please check your input.")
