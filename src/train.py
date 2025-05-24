import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Paths
PROCESSED_DIR = "./data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(PROCESSED_DIR, "model.joblib")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Trained model saved to: {MODEL_PATH}")

def main():
    print("🚀 Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("🧠 Training model...")
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
