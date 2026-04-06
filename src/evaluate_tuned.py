import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def evaluate_tuned():
    data_path = os.path.join(DATA_DIR, "processed_data.npz")
    model_path = os.path.join(MODELS_DIR, "best_ann_model_tuned.keras")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Processed data or tuned model not found.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]
    X_test21 = data["X_test21"] if "X_test21" in data else None
    y_test21 = data["y_test21"] if "y_test21" in data else None

    model = load_model(model_path, compile=False)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n--- Tuned Model Report (KDDTest+) ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    if X_test21 is not None and y_test21 is not None:
        print("\n--- Tuned Model Report (KDDTest-21) ---")
        y_pred21 = np.argmax(model.predict(X_test21), axis=1)
        print(classification_report(y_test21, y_pred21, target_names=classes))


if __name__ == "__main__":
    evaluate_tuned()
