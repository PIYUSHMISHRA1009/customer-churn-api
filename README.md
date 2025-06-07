# 💼 Customer Churn Prediction API

This project is an end-to-end **Machine Learning deployment pipeline** for predicting whether a customer is likely to churn. 
It includes data preprocessing, model training, containerization with Docker, and deployment to a cloud service via Render.

---

## 📊 Problem Statement

Customer churn is one of the biggest problems for businesses in retaining clients.
The goal is to build an API that can **predict whether a customer will churn** based on certain 
features like tenure, monthly charges, internet service, contract type, and more.

---

## 🏗️ Project Structure

```plaintext
mlops-customer-churn/
│
├── app/                        # Application code
│   ├── main.py                # Flask API app with /predict route
│   ├── model.py               # Loads model and preprocessor
│   └── utils.py               # Helper functions for predictions
│
├── data/                      # Data and artifacts
│   └── processed/
│       ├── model.joblib       # Trained ML model
│       └── preprocessor.joblib  # Scikit-learn preprocessing pipeline
│
├── Dockerfile                 # Docker config for containerization
├── requirements.txt           # List of dependencies
├── .gitignore                 # Files and folders to exclude from Git
├── README.md                  # Project overview and instructions (this file)
└── (Optional) notebooks/      # Jupyter notebooks used during EDA/training
```

---

## ⚙️ Setup Instructions (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/PIYUSHMISHRA1009/customer-churn-api.git
cd customer-churn-api
```

### 2. Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API locally

```bash
python app/main.py
```

API will be available at: `http://127.0.0.1:5000/`

---

## 🐳 Docker Deployment

### 1. Build Docker image

```bash
docker build -t churn-api .
```

### 2. Run Docker container

```bash
docker run -p 5000:5000 churn-api
```

Now open: [http://localhost:5000](http://localhost:5000)

---

## ☁️ Cloud Deployment (Render)

**🔗 API Live At:**  
[https://customer-churn-api-ako5.onrender.com](https://customer-churn-api-ako5.onrender.com)

**🌐 Frontend UI:**  
[https://churn-frontend-ui.onrender.com](https://churn-frontend-ui.onrender.com)

Render picks up changes from GitHub and automatically rebuilds the container using the `Dockerfile`.

---

## 🔗 API Endpoint Usage

### Endpoint

```http
POST /predict
```

### Request Body (JSON)

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.35,
  "TotalCharges": 355.2
}
```

### Sample Response

```json
{
  "prediction": "No",
  "probability": {
    "Churn": 0.47,
    "No Churn": 0.53
  }
}
```

---

## 📚 Tech Stack

* Python 🐍
* Pandas, NumPy, Scikit-learn 📊
* Flask 🌐
* Docker 🐳
* Render (Cloud Deployment) ☁️
* Git & GitHub 🔧

---

## 🎓 What I Learned

* How to structure an ML project for production
* Building APIs with Flask
* Containerizing apps using Docker
* Hosting a model as an API with Render
* Git operations and resolving remote conflicts
* Preparing and cleaning data for machine learning

---

## ✅ Completed Phases

✔️ Data cleaning & feature engineering  
✔️ Model training & evaluation  
✔️ Saved model & preprocessor as `joblib` files  
✔️ Built Flask API (`/predict`)  
✔️ Dockerized the app  
✔️ Pushed code & artifacts to GitHub  
✔️ Successfully deployed API on Render  
✔️ Connected with a live frontend for predictions  
✔️ Tested with real-time predictions  

---

## 🚧 Next Steps (TODO)

* 📈 Add logging & monitoring for production  
* 🧪 Write unit tests and setup CI/CD with GitHub Actions  
* 📊 Track model performance over time (MLflow, DVC, or EvidentlyAI)  
* 📦 Package this project as a PyPI library  

---

## 🙋‍♂️ How You Can Use This

You can use this API by sending a POST request to:

```
https://customer-churn-api-ako5.onrender.com/predict
```

Or try the live web UI:

**👉 [https://churn-frontend-ui.onrender.com](https://churn-frontend-ui.onrender.com)**

Use tools like:

* Postman
* cURL
* Python `requests` library

---

## 💬 Feedback

Open to contributions, suggestions, or feedback. Fork the repo and feel free to explore!
