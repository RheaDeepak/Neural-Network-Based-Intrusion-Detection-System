import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def train_model(use_class_weight=True):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # Load data
    data_path = os.path.join(DATA_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}. Please run preprocess.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    classes = data['classes']
    
    input_dim = X_train.shape[1]
    num_classes = len(classes)
    
    print(f"Building model (Input dim: {input_dim}, Num classes: {num_classes})...")
    model = build_model(input_dim, num_classes)
    
    model_path = os.path.join(MODELS_DIR, "best_ann_model.keras")
    
    # Callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    class_weight = None
    if use_class_weight:
        print("Computing class weights (enabled by default)...")
        unique_classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight = {int(c): float(w) for c, w in zip(unique_classes, weights)}
        print(f"Using class_weight: {class_weight}")

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weight
    )

    # Plotting
    print("Saving training plots...")
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_history.png"))
    plt.close()
    
    print("Training completed. Best model saved.")

if __name__ == "__main__":
    train_model()
