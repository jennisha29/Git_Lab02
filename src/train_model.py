import os, json, random, datetime
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

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
        n_samples = int(len(df))

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

    # Preprocess for real dataset (categoricals + numerics)
    if data_used == "real_csv":
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # Preview metrics (useful info for meta)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    model_version = f"model_{timestamp}"
    model_filename = f"models/{model_version}_lr_model.joblib"
    dump(model, model_filename)

    meta = {
        "timestamp": timestamp,
        "model_type": "LogisticRegression",
        "data_used": data_used,
        "n_samples": int(n_samples),
        "n_features": int(X_train.shape[1]) if data_used == "real_csv" else int(np.array(X).shape[1]),
        "accuracy_preview": acc,
        "f1_preview": f1,
        "model_path": model_filename,
        "meta_path": f"metrics/{timestamp}_train_meta.json",
    }

    with open(f"metrics/{timestamp}_train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
