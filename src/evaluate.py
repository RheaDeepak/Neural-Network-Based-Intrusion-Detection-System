import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def evaluate_and_alert():
    data_path = os.path.join(DATA_DIR, "processed_data.npz")
    model_path = os.path.join(MODELS_DIR, "best_ann_model.keras")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Required files not found. Ensure preprocess and train scripts have run.")
        return

    # Load data
    data = np.load(data_path, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    classes = data['classes']

    # Load model
    print("Loading model for evaluation...")
    model = load_model(model_path)
    
    # Predict
    print("Making predictions on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    print("\n--- ROC-AUC (One-vs-Rest) ---")
    num_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
    overall_auc = roc_auc_score(y_test_bin, y_pred_probs, average='macro', multi_class='ovr')
    print(f"Macro ROC-AUC (OvR): {overall_auc:.4f}")

    for i, cls in enumerate(classes):
        class_auc = roc_auc_score(y_test_bin[:, i], y_pred_probs[:, i])
        print(f"  {cls}: {class_auc:.4f}")

    print("Generating Confusion Matrix plot...")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    print("\n--- Simulating Real-time Alert Mechanism ---")
    # Identify indices of predictions that are classified as Attack (i.e. not 'Normal')
    normal_idx = list(classes).index('normal') if 'normal' in classes else -1
    
    alert_count = 0
    max_alerts_to_show = 10
    
    for i, pred in enumerate(y_pred):
        if pred != normal_idx:
            attack_type = classes[pred]
            true_type = classes[y_test[i]]
            print(f"[ALERT] Intrusion Detected! Type: {attack_type} (True label: {true_type})")
            alert_count += 1
        
        if alert_count >= max_alerts_to_show:
            print(f"... and many more. Suppressing further alerts.")
            break
            
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate_and_alert()
