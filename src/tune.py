import os
import numpy as np
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_tunable_model(hp, input_dim, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i in range(hp.Int("layers", 2, 3)):
        units = hp.Choice(f"units_{i}", values=[64, 96, 128, 160, 192])
        dropout = hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.5, step=0.1)
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation="softmax"))

    lr = hp.Float("lr", min_value=1e-4, max_value=3e-3, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    data_path = os.path.join(DATA_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        print("Processed data not found. Run preprocess.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = data["classes"]

    input_dim = X_train.shape[1]
    num_classes = len(classes)

    tuner = kt.RandomSearch(
        lambda hp: build_tunable_model(hp, input_dim, num_classes),
        objective="val_accuracy",
        max_trials=8,
        executions_per_trial=1,
        directory=MODELS_DIR,
        project_name="ann_tuner",
        overwrite=True,
    )

    tuner.search(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=12,
        batch_size=256,
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters:")
    for k in best_hp.values:
        print(f"  {k}: {best_hp.get(k)}")

    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=12,
        batch_size=256,
        verbose=1,
    )

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, "best_ann_model_tuned.keras")
    best_model.save(model_path)
    print(f"Saved tuned model to {model_path}")


if __name__ == "__main__":
    main()
