from pathlib import Path
from typing import Optional

import joblib


ML_FEATURE_NAMES = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_assets",
    "debt_to_equity",
    "roa",
    "operating_margin",
    "net_profit_margin",
    "asset_turnover",
    "working_capital_to_assets",
    "retained_earnings_to_assets",
    "ebit_to_assets",
    "equity_to_liabilities",
    "sales_to_assets",
]


MODEL_PATH = Path("models/kapitalac_ml_model.joblib")
_MODEL_CACHE = None


def safe_value(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def prepare_ml_features(scoring: dict) -> dict:
    ratios = scoring.get("ratios", {})
    variables = scoring.get("variables", {})

    features = {
        "current_ratio": safe_value(ratios.get("current_ratio")),
        "quick_ratio": safe_value(ratios.get("quick_ratio")),
        "cash_ratio": safe_value(ratios.get("cash_ratio")),
        "debt_to_assets": safe_value(ratios.get("debt_to_assets")),
        "debt_to_equity": safe_value(ratios.get("debt_to_equity")),
        "roa": safe_value(ratios.get("roa")),
        "operating_margin": safe_value(ratios.get("operating_margin")),
        "net_profit_margin": safe_value(ratios.get("net_profit_margin")),
        "asset_turnover": safe_value(ratios.get("asset_turnover")),
        "working_capital_to_assets": safe_value(
            variables.get("x1_working_capital_to_assets")
        ),
        "retained_earnings_to_assets": safe_value(
            variables.get("x2_retained_earnings_to_assets")
        ),
        "ebit_to_assets": safe_value(variables.get("x3_ebit_to_assets")),
        "equity_to_liabilities": safe_value(
            variables.get("x4_equity_to_liabilities")
        ),
        "sales_to_assets": safe_value(variables.get("x5_sales_to_assets")),
    }

    return features


def load_ml_model():
    """
    Učitava istrenirani ML model ako postoji.
    Ako fajl ne postoji, vraća None i sistem koristi rule-based fallback.
    """
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        return None

    _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def rule_based_risk_probability(scoring: dict, validation: dict) -> float:
    """
    Privremeni rule-based model.
    Koristi se samo dok models/kapitalac_ml_model.joblib ne postoji.
    """
    altman_private = scoring.get("altman_private", {})
    classification = altman_private.get("classification", {})
    risk_level = classification.get("risk_level")

    if risk_level == "low":
        probability = 0.12
    elif risk_level == "medium":
        probability = 0.45
    elif risk_level == "high":
        probability = 0.78
    else:
        probability = 0.50

    ratios = scoring.get("ratios", {})

    current_ratio = ratios.get("current_ratio")
    debt_to_assets = ratios.get("debt_to_assets")
    roa = ratios.get("roa")
    net_profit_margin = ratios.get("net_profit_margin")

    if current_ratio is not None and current_ratio < 1:
        probability += 0.10

    if debt_to_assets is not None and debt_to_assets > 0.7:
        probability += 0.12

    if roa is not None and roa < 0:
        probability += 0.12

    if net_profit_margin is not None and net_profit_margin < 0:
        probability += 0.10

    quality = validation.get("quality", {})
    quality_score = quality.get("score")

    if quality_score is not None and quality_score < 70:
        probability += 0.05

    probability = max(0.01, min(probability, 0.99))

    return round(probability, 4)


def classify_ml_probability(probability: float) -> dict:
    if probability < 0.25:
        return {
            "risk_class": "Low risk",
            "risk_label": "Nizak rizik",
            "risk_level": "low",
        }

    if probability < 0.60:
        return {
            "risk_class": "Medium risk",
            "risk_label": "Srednji rizik",
            "risk_level": "medium",
        }

    return {
        "risk_class": "High risk",
        "risk_label": "Visok rizik",
        "risk_level": "high",
    }


def predict_with_real_model(features: dict, model_bundle: dict) -> dict:
    """
    Predikcija pomoću pravog istreniranog modela.
    """
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    row = [[features.get(feature_name, 0.0) for feature_name in feature_names]]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])
    else:
        prediction = int(model.predict(row)[0])
        probability = 0.75 if prediction == 1 else 0.25

    probability = round(max(0.01, min(probability, 0.99)), 4)
    classification = classify_ml_probability(probability)

    return {
        "model_name": model_bundle.get("model_name", "Kapitalac ML Model"),
        "model_type": model_bundle.get("model_type", "machine_learning"),
        "is_ml_model_loaded": True,
        "risk_probability": probability,
        "risk_probability_percent": f"{probability * 100:.2f}%",
        "classification": classification,
        "features": features,
        "model_metrics": model_bundle.get("metrics", {}),
        "note": "Rezultat je izračunat pomoću istreniranog ML modela.",
    }


def predict_ml_risk(scoring: dict, validation: dict) -> dict:
    """
    Glavna funkcija za predikciju rizika.

    Ako postoji models/kapitalac_ml_model.joblib, koristi pravi ML model.
    Ako ne postoji, koristi privremeni rule-based model.
    """
    features = prepare_ml_features(scoring)
    model_bundle = load_ml_model()

    if model_bundle is not None:
        return predict_with_real_model(features=features, model_bundle=model_bundle)

    probability = rule_based_risk_probability(scoring, validation)
    classification = classify_ml_probability(probability)

    return {
        "model_name": "Kapitalac Risk Engine v0",
        "model_type": "rule_based_placeholder",
        "is_ml_model_loaded": False,
        "risk_probability": probability,
        "risk_probability_percent": f"{probability * 100:.2f}%",
        "classification": classification,
        "features": features,
        "note": (
            "Ovo je privremeni rule-based rezultat. "
            "Kada se istrenira pravi model i sačuva u models/kapitalac_ml_model.joblib, "
            "API će automatski koristiti ML model."
        ),
    }