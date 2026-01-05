# src/models/evaluate.py

import pandas as pd
import mlflow

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ------------------
# CONFIG
# ------------------
MLFLOW_TRACKING_URI = "sqlite:///mlruns_db/mlflow.db"


# ------------------
# EVALUATION FUNCTION
# ------------------
def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: str
) -> None:
    """
    Evaluate a trained model on test data and log metrics to MLflow.

    Args:
        model   : trained sklearn model
        X_test  : test features
        y_test  : test labels
        run_id  : MLflow run ID (same run used during training)
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("📊 Evaluating model...")

    # Attach to the same MLflow run
    with mlflow.start_run(run_id=run_id):

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("roc_auc", roc_auc)

        # Console output
        print("\n===== Classification Report =====")
        print(classification_report(y_test, y_pred))

        print("\n===== Confusion Matrix =====")
        print(confusion_matrix(y_test, y_pred))

        print(f"\n🔥 ROC-AUC Score: {roc_auc:.4f}")
