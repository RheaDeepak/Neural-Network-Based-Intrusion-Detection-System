import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Attack categories mapping
DOS = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
PROBE = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
R2L = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop', 'httptunnel']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']

def map_attack(label):
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
        # Fallback to general attack if unknown
        return 'Unknown_Attack'

def preprocess_data():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH, names=COMPREHENSIVE_COLS)
    test_df = pd.read_csv(TEST_PATH, names=COMPREHENSIVE_COLS)

    # Combine for consistent encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Drop difficulty as it's not a real feature
    combined_df.drop('difficulty', axis=1, inplace=True)

    # Map labels
    combined_df['mapped_label'] = combined_df['label'].apply(map_attack)
    combined_df.drop('label', axis=1, inplace=True)

    # Separate features and labels
    y_full = combined_df['mapped_label'].values
    combined_df.drop('mapped_label', axis=1, inplace=True)

    # Categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [col for col in combined_df.columns if col not in cat_cols and col != 'is_train']

    print("Encoding and scaling features...")
    # Setup standard scaler for numeric features and one-hot encoder for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])

    X_full = preprocessor.fit_transform(combined_df)
    
    # Label Encoding for targets
    le = LabelEncoder()
    y_full_encoded = le.fit_transform(y_full)
    print(f"Classes mapped: {le.classes_}")

    # Split back to train and test
    is_train_mask = combined_df['is_train'] == 1
    
    X_train = X_full[is_train_mask]
    y_train = y_full_encoded[is_train_mask]
    
    X_test = X_full[~is_train_mask]
    y_test = y_full_encoded[~is_train_mask]

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Save processed data
    print("Saving processed data...")
    np.savez(os.path.join(DATA_DIR, "processed_data.npz"), 
             X_train=X_train, y_train=y_train, 
             X_test=X_test, y_test=y_test,
             classes=le.classes_)
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()
