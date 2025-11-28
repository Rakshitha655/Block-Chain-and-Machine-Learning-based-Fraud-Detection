# ml_train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "creditcard.csv"   # must be present in project folder
TARGET_COL = "Class"           # Kaggle credit card fraud dataset uses 'Class' column

def load_and_prepare(path):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")
    # Basic preprocessing: drop Time, keep Amount and V1..V28
    X = df.drop([TARGET_COL, 'Time'], axis=1)
    y = df[TARGET_COL].astype(int)
    return X, y

def train_and_save():
    print("Loading dataset:", DATA_PATH)
    X, y = load_and_prepare(DATA_PATH)
    print("Using target:", TARGET_COL)
    print("Num samples:", X.shape[0], "Num features:", X.shape[1])

    # simple train/test split for model building
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Quick evaluation
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}")

    # Save model and feature columns
    joblib.dump({"model": clf, "columns": X.columns.tolist()}, "rf_model.joblib")
    print("Saved model to rf_model.joblib")

if __name__ == "__main__":
    train_and_save()
