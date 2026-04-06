import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

# Standard column names for NSL-KDD
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

# Attack categories mapping (5 classes only)
DOS = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
PROBE = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
R2L = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop', 'httptunnel']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']

def map_attack(label):
    if label == 'normal':
        return 'normal'
    elif label in DOS:
        return 'dos'
    elif label in PROBE:
        return 'probe'
    elif label in R2L:
        return 'r2l'
    elif label in U2R:
        return 'u2r'
    else:
        # Fallback: keep 5-class requirement by grouping unknown attacks into r2l
        return 'r2l'

def preprocess_data(imbalance_method='smote', random_state=42):
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH, names=COMPREHENSIVE_COLS)
    test_df = pd.read_csv(TEST_PATH, names=COMPREHENSIVE_COLS)

    # Combine and perform stratified split as requested
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Drop difficulty as it's not a real feature
    combined_df.drop('difficulty', axis=1, inplace=True)

    # Map labels
    combined_df['mapped_label'] = combined_df['label'].apply(map_attack)
    y = combined_df['mapped_label'].values
    X_df = combined_df.drop(['label', 'mapped_label'], axis=1)

    # Categorical and numerical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [col for col in X_df.columns if col not in cat_cols]

    # Label encode targets
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes mapped: {le.classes_}")

    # Stratified split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y_encoded
    )

    print("Encoding and scaling features...")
    # Fit transform on train and transform test
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    if imbalance_method.lower() == 'smote':
        print("Applying SMOTE for multi-class imbalance handling...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Save processed data
    print("Saving processed data and metadata...")
    np.savez(os.path.join(DATA_DIR, "processed_data.npz"), 
             X_train=X_train, y_train=y_train, 
             X_test=X_test, y_test=y_test,
             classes=le.classes_,
             cat_cols=np.array(cat_cols, dtype=object),
             num_cols=np.array(num_cols, dtype=object),
             imbalance_method=imbalance_method)
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()
