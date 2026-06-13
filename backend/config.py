import os


APP_NAME = "Kapitalac API"
APP_VERSION = "0.7.0"

# Dozvoljeni domeni sa kojih frontend može da poziva backend.
# U produkciji je preporučeno podesiti ALLOWED_ORIGINS preko Render environment varijabli.
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")

if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in ALLOWED_ORIGINS_RAW.split(",")
        if origin.strip()
    ]

# Maksimalna dozvoljena veličina PDF fajla.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# MLflow tracking za lokalno praćenje eksperimenata.
# Model se ne čuva kroz MLflow registry, već kao joblib fajl u models/ folderu.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Nove verzije MLflow-a traže eksplicitnu dozvolu za lokalni file-store način rada.
MLFLOW_ALLOW_FILE_STORE = os.getenv("MLFLOW_ALLOW_FILE_STORE", "true")