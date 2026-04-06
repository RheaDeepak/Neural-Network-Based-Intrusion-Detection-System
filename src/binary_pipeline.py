import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

COMPREHENSIVE_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]


def load_dataset():
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    train_df = pd.read_csv(train_path, names=COMPREHENSIVE_COLS)
    test_df = pd.read_csv(test_path, names=COMPREHENSIVE_COLS)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df.drop("difficulty", axis=1, inplace=True)
    return df


def preprocess(df):
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
    y = df["label"].values
    X = df.drop("label", axis=1)

    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_roc(y_true, y_prob, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary IDS)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return auc_score


def main(use_class_weight: bool, threshold: float):
    df = load_dataset()
    X, y, preprocessor = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if not use_class_weight:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    model = build_model(X_train.shape[1])

    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): w for c, w in zip(classes, weights)}

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    print("\n--- Binary Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    auc_score = plot_roc(y_test, y_prob, os.path.join(PLOTS_DIR, "roc_curve_binary.png"))
    print(f"ROC AUC: {auc_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-class-weight", action="store_true", help="Use class weights instead of SMOTE")
    parser.add_argument("--threshold", type=float, default=0.3, help="Classification threshold")
    args = parser.parse_args()
    main(args.use_class_weight, args.threshold)
