# src/models/train.py

from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# ========================
# CONFIG
# ========================
EXPERIMENT_NAME = "credit_fraud"
MODEL_NAME = "fraud_detection"
MLFLOW_TRACKING_URI = "sqlite:///mlruns_db/mlflow.db"
PRODUCTION_ALIAS = "production"


# ========================
# TRAIN FUNCTION
# ========================
def train_model(df: pd.DataFrame) -> Tuple[
    RandomForestClassifier, pd.DataFrame, pd.Series, str
]:
    """
    Train a RandomForest model, log metrics to MLflow, 
    register it in the MLflow Model Registry, and set alias 'production'.

    Args:
        df (pd.DataFrame): Preprocessed dataset including target column 'Class'

    Returns:
        clf       : trained RandomForestClassifier
        X_test    : test features
        y_test    : test labels
        run_id    : MLflow run ID
    """

    # -----------------------
    # Split features and target
    # -----------------------
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------
    # Initialize model
    # -----------------------
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # -----------------------
    # MLflow setup
    # -----------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # -----------------------
    # Train & log
    # -----------------------
    with mlflow.start_run() as run:
        clf.fit(X_train, y_train)

        # Metrics
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        # -----------------------
        # Log & register model
        # -----------------------
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="fraud_model",
            registered_model_name=MODEL_NAME
        )

        # Retrieve registered model version
        versions = client.get_latest_versions(MODEL_NAME)
        latest_version = max([v.version for v in versions], key=int)

        # Assign alias 'production'
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=PRODUCTION_ALIAS,
            version=latest_version
        )

        print("✅ Model trained, registered, and aliased to 'production'")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy : {test_acc:.4f}")
        print(f"   Run ID        : {run.info.run_id}")
        print(f"   Registered Version: {latest_version}")

    return clf, X_test, y_test, run.info.run_id
