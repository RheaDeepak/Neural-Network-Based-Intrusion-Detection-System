import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")
TEST21_PATH = os.path.join(DATA_DIR, "KDDTest-21.txt")

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


def map_binary(label: str) -> str:
    return "Normal" if label == "normal" else "Attack"


def preprocess_binary():
    train_df = pd.read_csv(TRAIN_PATH, names=COMPREHENSIVE_COLS)
    test_df = pd.read_csv(TEST_PATH, names=COMPREHENSIVE_COLS)
    test21_df = pd.read_csv(TEST21_PATH, names=COMPREHENSIVE_COLS) if os.path.exists(TEST21_PATH) else None

    train_df["is_train"] = 1
    test_df["is_train"] = 0
    if test21_df is not None:
        test21_df["is_train"] = 0
        test21_df["is_test21"] = 1

    combined_df = pd.concat([train_df, test_df] + ([test21_df] if test21_df is not None else []), ignore_index=True)
    combined_df.drop("difficulty", axis=1, inplace=True)

    combined_df["mapped_label"] = combined_df["label"].apply(map_binary)
    combined_df.drop("label", axis=1, inplace=True)

    y_full = combined_df["mapped_label"].values
    combined_df.drop("mapped_label", axis=1, inplace=True)

    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [col for col in combined_df.columns if col not in cat_cols and col not in ("is_train", "is_test21")]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ]
    )

    X_full = preprocessor.fit_transform(combined_df)
    le = LabelEncoder()
    y_full_encoded = le.fit_transform(y_full)

    is_train_mask = combined_df["is_train"] == 1
    X_train = X_full[is_train_mask]
    y_train = y_full_encoded[is_train_mask]
    X_test = X_full[~is_train_mask]
    y_test = y_full_encoded[~is_train_mask]

    X_test21, y_test21 = None, None
    if "is_test21" in combined_df.columns:
        is_test21_mask = combined_df["is_test21"] == 1
        X_test21 = X_full[is_test21_mask]
        y_test21 = y_full_encoded[is_test21_mask]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    top_k = max(20, int(0.8 * X_train.shape[1]))
    selected_indices = np.argsort(mi_scores)[::-1][:top_k]
    X_train = X_train[:, selected_indices]
    X_test = X_test[:, selected_indices]
    if X_test21 is not None:
        X_test21 = X_test21[:, selected_indices]

    pca = PCA(n_components=0.95, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    if X_test21 is not None:
        X_test21 = pca.transform(X_test21)

    np.savez(
        os.path.join(DATA_DIR, "processed_data_binary.npz"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_test21=X_test21,
        y_test21=y_test21,
        classes=le.classes_,
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "selected_indices": selected_indices,
            "pca": pca,
            "classes": le.classes_,
        },
        os.path.join(DATA_DIR, "preprocess_artifacts_binary.joblib"),
    )

    print("Binary preprocessing complete.")


if __name__ == "__main__":
    preprocess_binary()
