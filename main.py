# main.py

from pathlib import Path
import subprocess

from src.data.preprocessing import load_data, preprocess_data, save_preprocess
from src.models.train import train_model
from src.models.evaluate import evaluate

from mlflow.tracking import MlflowClient
import mlflow

# ------------------
# CONFIG / PATHS
# ------------------
RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_PATH = Path("data/processed/preprocess_creditcard.csv")
MODEL_NAME = "fraud_detection"
MLFLOW_TRACKING_URI = "sqlite:///mlruns_db/mlflow.db"
PRODUCTION_ALIAS = "production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ------------------
# 1️⃣ Preprocess
# ------------------
print("📥 Loading raw dataset...")
df_raw = load_data(RAW_PATH)

print("⚙️ Preprocessing dataset...")
df_processed = preprocess_data(df_raw)

print("💾 Saving processed dataset...")
save_preprocess(df_processed, PROCESSED_PATH)

# ------------------
# 2️⃣ DVC Tracking (no git commit)
# ------------------
print("📦 Tracking processed data with DVC...")
try:
    subprocess.run(["dvc", "add", str(PROCESSED_PATH)], check=True)
except subprocess.CalledProcessError:
    print("ℹ DVC: Data already tracked")

# ------------------
# 3️⃣ Train + Register
# ------------------
print("🤖 Training model...")
clf, X_test, y_test, run_id = train_model(df_processed)
print(f"🔎 MLflow Run ID: {run_id}")

# ------------------
# 4️⃣ Promote Latest Version to Production Alias
# ------------------
print("🚀 Promoting latest model to production alias...")
client = MlflowClient()

# Get latest registered version
latest_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1].version

# Set alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=PRODUCTION_ALIAS,
    version=latest_version
)
print(f"✅ Model version {latest_version} is now @{PRODUCTION_ALIAS}")

# ------------------
# 5️⃣ Evaluate & Log
# ------------------
print("📊 Evaluating...")
evaluate(clf, X_test, y_test, run_id)

# ------------------
# 6️⃣ Registry Status
# ------------------
versions = client.get_latest_versions(MODEL_NAME)
print(f"\n📚 Model Registry: {MODEL_NAME}")
for v in versions:
    print(f" - Version {v.version} | Alias: {v.aliases}")

print("\n🎉 Pipeline completed successfully!")
print(f"➡️ Start API: uvicorn src.api.api:app --reload")
