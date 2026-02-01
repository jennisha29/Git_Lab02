import os, json, random
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    model_path = f"models/model_{timestamp}_lr_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    dataset_path = "data/student_lifestyle_100k.csv"

    # Use REAL dataset if available, else SYNTHETIC fallback
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)

        if "Depression" not in df.columns:
            raise ValueError("Expected target column 'Depression' not found in dataset.")

        if "Student_ID" in df.columns:
            df = df.drop(columns=["Student_ID"])

        y = df["Depression"].astype(int)
        X = df.drop(columns=["Depression"])
        data_used = "real_csv"

    else:
        n_samples = random.randint(200, 2000)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=6,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            random_state=0,
            shuffle=True,
        )
        data_used = "synthetic_fallback"

    # train/test split (stratify only if valid)
    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=stratify
    )

    y_pred = model.predict(X_test)

    metrics = {
        "timestamp": timestamp,
        "model_type": "LogisticRegression",
        "data_used": data_used,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_test": int(len(y_test)),
        "model_path": model_path,
        "metrics_path": f"metrics/{timestamp}_metrics.json",
    }

    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{timestamp}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)