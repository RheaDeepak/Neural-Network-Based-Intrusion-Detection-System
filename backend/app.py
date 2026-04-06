from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_ann_model.keras")

# Columns (same as in preprocess.py)
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

cat_cols = ['protocol_type', 'service', 'flag']


class Sample(BaseModel):
    sample: Dict[str, Any]


class BatchSamples(BaseModel):
    samples: List[Dict[str, Any]]


app = FastAPI(title="Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def map_attack(label: str) -> str:
    DOS = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    PROBE = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    R2L = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop', 'httptunnel']
    U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    if label == 'normal':
        return 'Normal'
    elif label in DOS:
        return 'DoS'
    elif label in PROBE:
        return 'Probe'
    elif label in R2L:
        return 'R2L'
    elif label in U2R:
        return 'U2R'
    else:
        return 'Normal'


# Globals populated at startup
preprocessor = None
num_cols = None
le = None
model = None
classes = None
insights_cache = None
selected_indices = None
pca = None


def startup_processing():
    global preprocessor, num_cols, le, model, classes, insights_cache, selected_indices, pca

    # Load raw datasets and fit preprocessor and label encoder so we can handle unseen inputs
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise RuntimeError("Training or test data files are missing in the data/ directory.")

    train_df = pd.read_csv(TRAIN_PATH, names=COMPREHENSIVE_COLS)
    test_df = pd.read_csv(TEST_PATH, names=COMPREHENSIVE_COLS)

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.drop('difficulty', axis=1, inplace=True)

    # Map labels and keep for label encoder
    combined_df['mapped_label'] = combined_df['label'].apply(map_attack)
    le = LabelEncoder()
    le.fit(combined_df['mapped_label'].values)
    classes = list(le.classes_)

    # Determine numeric columns (exclude is_train)
    num_cols = [col for col in combined_df.columns if col not in cat_cols and col not in ('is_train', 'label', 'mapped_label')]

    artifacts_path = os.path.join(DATA_DIR, "preprocess_artifacts.joblib")
    if os.path.exists(artifacts_path):
        artifacts = joblib.load(artifacts_path)
        preprocessor = artifacts["preprocessor"]
        selected_indices = artifacts["selected_indices"]
        pca = artifacts["pca"]
        classes = list(artifacts["classes"])
    else:
        # Fit preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), num_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ]
        )

        # Fit on the combined dataframe (drop columns not used)
        preprocessor.fit(combined_df.drop(['label', 'mapped_label'], axis=1))

        # Transform and compute MI + PCA on training data
        X_full = preprocessor.transform(combined_df.drop(['label', 'mapped_label'], axis=1))
        is_train_mask = combined_df['is_train'] == 1
        X_train = X_full[is_train_mask]
        y_train = le.transform(combined_df.loc[is_train_mask, 'mapped_label'])

        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        top_k = max(20, int(0.8 * X_train.shape[1]))
        selected_indices = np.argsort(mi_scores)[::-1][:top_k]
        X_train = X_train[:, selected_indices]

        pca = PCA(n_components=0.95, random_state=42)
        pca.fit(X_train)

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please train or place the Keras model there.")
    model = load_model(MODEL_PATH, compile=False)

    # Build insights
    mapped_labels = combined_df['mapped_label']
    label_counts = mapped_labels.value_counts().to_dict()
    train_counts = train_df['label'].apply(map_attack).value_counts().to_dict()
    test_counts = test_df['label'].apply(map_attack).value_counts().to_dict()
    total_samples = len(combined_df)

    # Model summary info
    param_count = int(model.count_params())
    input_dim = int(model.input_shape[-1]) if model and model.input_shape else None

    insights_cache = {
        "total_samples": total_samples,
        "label_distribution": label_counts,
        "train_label_distribution": train_counts,
        "test_label_distribution": test_counts,
        "num_features": input_dim,
        "num_classes": len(classes),
        "model_params": param_count
    }


@app.on_event("startup")
def on_startup():
    try:
        startup_processing()
        print("Backend startup: model and preprocessors loaded.")
    except Exception as e:
        print("Error during startup:", e)


def validate_and_df(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    # Ensure required keys present; build DataFrame with columns used by preprocessor
    expected_cols = num_cols + cat_cols
    rows = []
    for s in samples:
        row = {}
        for c in expected_cols:
            if c in s:
                row[c] = s[c]
            else:
                # If numeric missing, set 0. If categorical missing, set a placeholder
                if c in cat_cols:
                    row[c] = s.get(c, 'other')
                else:
                    row[c] = s.get(c, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


@app.get("/classes")
def get_classes():
    return {"classes": classes}


@app.get("/insights")
def get_insights():
    if insights_cache is None:
        raise HTTPException(status_code=500, detail="Insights are not ready yet.")
    return insights_cache


@app.post("/predict")
def predict(batch: BatchSamples):
    samples = batch.samples
    if not samples or not isinstance(samples, list):
        raise HTTPException(status_code=400, detail="Provide a 'samples' list of feature dictionaries.")

    try:
        df = validate_and_df(samples)
        X = preprocessor.transform(df)
        if selected_indices is not None:
            X = X[:, selected_indices]
        if pca is not None:
            X = pca.transform(X)
        preds_proba = model.predict(X)
        pred_idx = np.argmax(preds_proba, axis=1)
        pred_labels = [classes[i] for i in pred_idx]

        results = []
        for i, s in enumerate(samples):
            results.append({
                "input": s,
                "predicted_class": pred_labels[i],
                "probabilities": {classes[j]: float(preds_proba[i][j]) for j in range(len(classes))}
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
