import os
import tempfile

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.altman_score import calculate_altman_scores
from backend.config import (
    ALLOWED_ORIGINS,
    APP_NAME,
    APP_VERSION,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
)
from backend.ml_model import (
    ML_FEATURE_NAMES,
    MODEL_PATH,
    load_ml_model,
    predict_ml_risk,
)
from backend.pdf_extractor import extract_financials_from_pdf
from backend.validation import build_lovable_payload, validate_financial_data


app = FastAPI(
    title=APP_NAME,
    description="API za analizu crnogorskih finansijskih izvještaja i procjenu rizika poslovanja.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "app": "Kapitalac",
        "message": "Kapitalac API radi.",
        "status": "ok",
        "version": APP_VERSION,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kapitalac-api",
        "version": APP_VERSION,
    }


@app.get("/model-status")
def model_status():
    model_bundle = load_ml_model()

    if model_bundle is None:
        return {
            "status": "ok",
            "is_ml_model_loaded": False,
            "model_file_exists": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "active_model": {
                "model_name": "Kapitalac Risk Engine v0",
                "model_type": "rule_based_placeholder",
            },
            "message": "Pravi ML model nije pronađen. API koristi rule-based fallback.",
            "features": ML_FEATURE_NAMES,
        }

    return {
        "status": "ok",
        "is_ml_model_loaded": True,
        "model_file_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "active_model": {
            "model_name": model_bundle.get("model_name"),
            "model_type": model_bundle.get("model_type"),
            "metrics": model_bundle.get("metrics", {}),
        },
        "message": "API koristi istrenirani ML model.",
        "features": model_bundle.get("feature_names", ML_FEATURE_NAMES),
    }


@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Naziv fajla nije pronađen.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Dozvoljeni su samo PDF fajlovi.",
        )

    temp_file_path = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="PDF fajl je prazan.",
            )

        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF fajl je prevelik. Maksimalna dozvoljena veličina je {MAX_UPLOAD_SIZE_MB} MB.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(content)

        extracted_data = extract_financials_from_pdf(temp_file_path)
        extracted_data["file_name"] = file.filename

        validation = validate_financial_data(extracted_data)
        scoring = calculate_altman_scores(extracted_data)
        ml_prediction = predict_ml_risk(scoring=scoring, validation=validation)

        lovable_payload = build_lovable_payload(
            financials=extracted_data,
            scoring=scoring,
            validation=validation,
            ml_prediction=ml_prediction,
        )

        return {
            "status": "success",
            "message": "PDF je uspješno obrađen i analiziran.",
            "data": {
                "lovable": lovable_payload,
                "company": {
                    "file_name": extracted_data.get("file_name"),
                    "company_name": extracted_data.get("company_name"),
                    "registration_number": extracted_data.get("registration_number"),
                    "activity_code": extracted_data.get("activity_code"),
                    "report_year": extracted_data.get("report_year"),
                },
                "financials": extracted_data,
                "validation": validation,
                "scoring": scoring,
                "ml_prediction": ml_prediction,
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Greška prilikom obrade PDF-a: {str(error)}",
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/calculate-altman")
def calculate_altman(financials: dict = Body(...)):
    try:
        validation = validate_financial_data(financials)
        scoring = calculate_altman_scores(financials)
        ml_prediction = predict_ml_risk(scoring=scoring, validation=validation)

        lovable_payload = build_lovable_payload(
            financials=financials,
            scoring=scoring,
            validation=validation,
            ml_prediction=ml_prediction,
        )

        return {
            "status": "success",
            "message": "Altman Z-Score je uspješno izračunat.",
            "data": {
                "lovable": lovable_payload,
                "validation": validation,
                "scoring": scoring,
                "ml_prediction": ml_prediction,
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Greška prilikom računanja Altman score-a: {str(error)}",
        )


@app.post("/predict-risk")
def predict_risk(payload: dict = Body(...)):
    try:
        scoring = payload.get("scoring", {})
        validation = payload.get("validation", {})

        prediction = predict_ml_risk(scoring=scoring, validation=validation)

        return {
            "status": "success",
            "message": "Predikcija rizika je uspješno izračunata.",
            "data": prediction,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Greška prilikom predikcije rizika: {str(error)}",
        )


@app.get("/model-info")
def model_info():
    return {
        "app": "Kapitalac",
        "version": APP_VERSION,
        "available_models": [
            {
                "name": "Altman Z-Score",
                "type": "financial_formula",
                "purpose": "Baseline procjena rizika bankrota",
            },
            {
                "name": "Altman Z'-Score",
                "type": "financial_formula",
                "purpose": "Procjena rizika za privatne firme koje se ne kotiraju na berzi",
            },
            {
                "name": "Financial Ratio Analysis",
                "type": "financial_ratios",
                "purpose": "Likvidnost, zaduženost, profitabilnost i obrt aktive",
            },
            {
                "name": "Data Quality Validation",
                "type": "validation",
                "purpose": "Provjera da li su podaci iz PDF-a kompletni i konzistentni",
            },
            {
                "name": "Kapitalac ML Model",
                "type": "machine_learning",
                "purpose": "ML procjena rizika na osnovu finansijskih pokazatelja",
            },
        ],
        "ml_note": (
            "Trenutni ML model je weak-supervised model treniran na Altman Z' zonama. "
            "U produkciji bi se koristile stvarne istorijske labele kao što su stečaj, "
            "likvidacija, blokada računa ili neizmirene obaveze."
        ),
        "main_endpoint_for_lovable": "/analyze-pdf",
        "model_status_endpoint": "/model-status",
        "allowed_origins": ALLOWED_ORIGINS,
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
    }