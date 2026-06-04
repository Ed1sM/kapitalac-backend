import os


APP_NAME = "Kapitalac API"
APP_VERSION = "0.7.0"

ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")

if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in ALLOWED_ORIGINS_RAW.split(",")
        if origin.strip()
    ]

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Stabilno lokalno MLflow logovanje za projekat.
# MLflow zapisuje eksperimente u folder mlruns/.
# Model se ne čuva kroz MLflow registry, već preko joblib-a.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Potrebno za nove MLflow verzije koje file-store drže u maintenance režimu.
MLFLOW_ALLOW_FILE_STORE = os.getenv("MLFLOW_ALLOW_FILE_STORE", "true")