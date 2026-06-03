import os


APP_NAME = "Kapitalac API"
APP_VERSION = "0.6.0"

# Kasnije kada dobiješ Lovable URL, dodaćeš ga u environment varijablu:
# ALLOWED_ORIGINS=https://tvoj-lovable-sajt.lovable.app,https://kapitalac.me
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")

if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in ALLOWED_ORIGINS_RAW.split(",")
        if origin.strip()
    ]

# Maksimalna veličina PDF fajla: 15 MB
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024