import argparse, json, os, pickle, sys
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
sys.path.insert(0, os.path.abspath(".."))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    model_filename = f"models/model_{timestamp}_depression_rf_model.joblib"

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}. Make sure train_model.py ran successfully.")
    model = joblib.load(model_filename)
    print(f"Loaded model: {model_filename}")
    
    x_test_path = "data/X_test.pickle"
    y_test_path = "data/y_test.pickle"

    if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        raise FileNotFoundError("Test split not found. Expected data/X_test.pickle and data/y_test.pickle. Make sure train_model.py saved them.")
    
    with open(x_test_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_test_path, "rb") as f:
        y_test = pickle.load(f)
    
    y_pred = model.predict(X_test)
    metrics = {"F1_Score": float(f1_score(y_test, y_pred)), "Accuracy": float(accuracy_score(y_test, y_pred)), "Precision": float(precision_score(y_test, y_pred)), "Recall": float(recall_score(y_test, y_pred)), "timestamp": timestamp}
    
    os.makedirs("metrics", exist_ok=True)

    metrics_filename = f"metrics/{timestamp}_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics: {metrics_filename}")
    print(metrics)