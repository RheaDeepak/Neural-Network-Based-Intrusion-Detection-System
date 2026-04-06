import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def evaluate_binary():
    data_path = os.path.join(DATA_DIR, "processed_data_binary.npz")
    model_path = os.path.join(MODELS_DIR, "best_ann_model_binary.keras")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Binary processed data or model not found. Run preprocess_binary.py and train_binary.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]
    X_test21 = data["X_test21"] if "X_test21" in data else None
    y_test21 = data["y_test21"] if "y_test21" in data else None

    model = load_model(model_path, compile=False)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n--- Binary Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Binary Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_binary.png"))
    plt.close()

    if "Normal" in classes:
        normal_idx = list(classes).index("Normal")
        y_true_attack = (y_test != normal_idx)
        y_pred_attack = (y_pred != normal_idx)
        fp = np.sum((~y_true_attack) & y_pred_attack)
        tn = np.sum((~y_true_attack) & (~y_pred_attack))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"False Positive Rate (Normal as negative): {fpr:.4f}")

    if X_test21 is not None and y_test21 is not None:
        print("\n--- Hard Test (KDDTest-21) Binary Report ---")
        y_pred21 = np.argmax(model.predict(X_test21), axis=1)
        print(classification_report(y_test21, y_pred21, target_names=classes))


if __name__ == "__main__":
    evaluate_binary()
