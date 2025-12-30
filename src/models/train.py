import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn



MODEL_PATH= Path("models/model.pkl")
Preprocess_DATAPATH = Path("data/processed/process_data.csv")


def load_preprocess_data(path:Path=Preprocess_DATAPATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed Dataset not found at {path}")
    
    return pd.read_csv(path)

def train_model(df:pd.DataFrame):
    X=df.drop(columns=["Class"])
    y=df["Class"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

    clf=RandomForestClassifier(
        n_estimators= 200 ,
        class_weight= "balanced" ,     
        random_state= 42,
        n_jobs=-1
    )

    mlflow.set_experiment("credit_fraud")

    with mlflow.start_run():
        clf.fit(X_train,y_train)
        train_score= clf.score(X_train,y_train)
        test_score= clf.score(X_test,y_test)

        mlflow.log_param("n_estimator",200)
        mlflow.log_metric("training accuracy",train_score)
        mlflow.log_metric("testing metric", test_score)

        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy:  {test_score:.4f}")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf,MODEL_PATH)

        mlflow.sklearn.log_model(clf,"fraud_model")

    return clf, X_test, y_test


