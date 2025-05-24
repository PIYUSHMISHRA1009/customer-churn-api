import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
RAW_DATA_PATH = "./data/raw/customer_churn.csv"
PROCESSED_DIR = "./data/processed"
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def preprocess_and_save(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Identify columns
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    # Fit and transform
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save preprocessor
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(pipeline, PREPROCESSOR_PATH)

    # Convert to DataFrame and save
    X_train_df = pd.DataFrame(X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed)
    X_train_df["Churn"] = y_train.reset_index(drop=True)

    X_test_df = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed)
    X_test_df["Churn"] = y_test.reset_index(drop=True)

    X_train_df.to_csv(TRAIN_PATH, index=False)
    X_test_df.to_csv(TEST_PATH, index=False)

    print(f"Preprocessing complete.")
    print(f"Train data saved to: {TRAIN_PATH}")
    print(f"Test data saved to: {TEST_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    df = load_and_clean_data(RAW_DATA_PATH)
    preprocess_and_save(df)
