import argparse, datetime, os, pickle, sys
import mlflow, pandas as pd, numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
sys.path.insert(0, os.path.abspath(".."))

def find_dataset_path():
    candidates = ["data/student_lifestyle_100k.csv", "student_lifestyle_100k.csv"]
    env_path = os.getenv("DATASET_PATH")

    if env_path:
        candidates.insert(0, env_path)
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("student_lifestyle_100k.csv not found. Tried: " + ", ".join(candidates) + ". Put it in data/ or set DATASET_PATH.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)
    required_cols = {"Student_ID", "Depression", "Gender", "Department"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    
    X = df.drop(["Student_ID", "Depression"], axis=1)
    y = df["Depression"].astype(int)
    
    label_encoders = {}
    for col in ["Gender", "Department"]:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs("data", exist_ok=True)
    with open("data/X_test.pickle", "wb") as f:
        pickle.dump(X_test, f)
    with open("data/y_test.pickle", "wb") as f:
        pickle.dump(y_test, f)
    with open("data/label_encoders.pickle", "wb") as f:
        pickle.dump(label_encoders, f)
    with open("data/scaler.pickle", "wb") as f:
        pickle.dump(scaler, f)
    
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Student_Lifestyle_Depression"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        mlflow.log_params({"dataset_name": dataset_name, "rows": int(df.shape[0]), "features": int(X.shape[1]), "test_size": 0.2, "random_state": 42, "model": "RandomForestClassifier", "n_estimators": 300, "max_depth": 20, "class_weight": "balanced", "scaled": "yes"})
        
        forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        forest.fit(X_train, y_train)
        
        y_pred = forest.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metrics({"Accuracy": float(acc), "F1 Score": float(f1)})
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/model_{timestamp}_depression_rf_model.joblib"
    dump(forest, model_filename)
    print(f"Saved model: {model_filename}")