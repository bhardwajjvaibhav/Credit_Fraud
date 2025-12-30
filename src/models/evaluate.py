import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score)

MODEL_PATH = Path("models/model.pkl")
Preprocess_Datapath = Path("data/processed/preprocess_creditcard.csv")


def load_data(path:Path=Preprocess_Datapath):
    if not path.exists():
        raise FileNotFoundError
    
    return pd.read_csv(path)


def load_model(path:Path=MODEL_PATH):
    if not path.exists():
        raise ModuleNotFoundError
    
    return joblib.load(MODEL_PATH)


def evaluate():

    print("EVALUATING MODEL........")
    df=load_data()
    model=load_model()
    X_test = df.drop(columns=["Class"])
    y_test = df["Class"]

    print("🔍 Running evaluation...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {auc_score:.4f}")



