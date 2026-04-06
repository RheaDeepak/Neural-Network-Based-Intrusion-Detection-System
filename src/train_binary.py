import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def train_binary():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    data_path = os.path.join(DATA_DIR, "processed_data_binary.npz")
    if not os.path.exists(data_path):
        print("Binary processed data not found. Run preprocess_binary.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = data["classes"]

    input_dim = X_train.shape[1]
    num_classes = len(classes)

    print(f"Building binary model (Input dim: {input_dim}, Num classes: {num_classes})...")
    model = build_model(input_dim, num_classes)

    model_path = os.path.join(MODELS_DIR, "best_ann_model_binary.keras")

    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=256,
        callbacks=[checkpoint, early_stop],
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Binary Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Binary Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_history_binary.png"))
    plt.close()

    print("Binary training completed. Best model saved.")


if __name__ == "__main__":
    train_binary()
