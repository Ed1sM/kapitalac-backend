from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


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
    """
    Vraća numeričku vrijednost za ML model.

    Ako podatak nedostaje ili nije moguće pretvoriti ga u broj,
    koristi se podrazumijevana vrijednost.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def prepare_ml_features(scoring: dict) -> dict:
    """
    Priprema ulazne finansijske pokazatelje za ML model.
    """
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
    Učitava sačuvani ML model.

    Model se kešira nakon prvog učitavanja kako se ne bi čitao sa diska
    pri svakom API pozivu.
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
    Privremena rule-based procjena rizika.

    Koristi se samo kada pravi ML model nije pronađen u models/ folderu.
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
    """
    Pretvara vjerovatnoću rizika u poslovnu klasifikaciju.
    """
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


def make_feature_row(features: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Pravi jedan red podataka u formatu koji očekuje scikit-learn pipeline.
    """
    return pd.DataFrame(
        [[features.get(feature_name, 0.0) for feature_name in feature_names]],
        columns=feature_names,
    )


def predict_probability_for_model(model, features: dict, feature_names: list[str]) -> float:
    """
    Računa vjerovatnoću rizika za jedan model.
    """
    row = make_feature_row(features=features, feature_names=feature_names)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])
    else:
        prediction = int(model.predict(row)[0])
        probability = 0.75 if prediction == 1 else 0.25

    probability = round(max(0.01, min(probability, 0.99)), 4)

    return probability


def build_model_prediction_item(
    model_name: str,
    probability: float,
    metrics: dict,
    active_model_name: str,
) -> dict:
    """
    Pravi standardizovan rezultat predikcije za jedan model.
    """
    classification = classify_ml_probability(probability)

    return {
        "model_name": model_name,
        "is_active": model_name == active_model_name,
        "risk_probability": probability,
        "risk_probability_percent": f"{probability * 100:.2f}%",
        "classification": classification,
        "risk_class": classification.get("risk_class"),
        "risk_label": classification.get("risk_label"),
        "risk_level": classification.get("risk_level"),
        "metrics": metrics or {},
    }


def build_candidate_predictions(
    model_bundle: dict,
    features: dict,
) -> list[dict]:
    """
    Pravi predikcije za sve sačuvane ML kandidate.

    Ovo omogućava da se na frontend-u prikaže poređenje više modela,
    a ne samo rezultat aktivnog modela.
    """
    feature_names = model_bundle.get("feature_names", ML_FEATURE_NAMES)
    active_model_name = model_bundle.get("model_name", "Kapitalac ML Model")
    all_model_metrics = model_bundle.get("all_model_metrics", {})

    candidate_models = model_bundle.get("candidate_models")

    predictions = []

    if isinstance(candidate_models, dict) and candidate_models:
        for model_name, model in candidate_models.items():
            try:
                probability = predict_probability_for_model(
                    model=model,
                    features=features,
                    feature_names=feature_names,
                )

                predictions.append(
                    build_model_prediction_item(
                        model_name=model_name,
                        probability=probability,
                        metrics=all_model_metrics.get(model_name, {}),
                        active_model_name=active_model_name,
                    )
                )
            except Exception as error:
                predictions.append(
                    {
                        "model_name": model_name,
                        "is_active": model_name == active_model_name,
                        "error": str(error),
                        "risk_probability": None,
                        "risk_probability_percent": None,
                        "classification": {
                            "risk_class": "Unavailable",
                            "risk_label": "Nedostupno",
                            "risk_level": "unknown",
                        },
                        "risk_class": "Unavailable",
                        "risk_label": "Nedostupno",
                        "risk_level": "unknown",
                        "metrics": all_model_metrics.get(model_name, {}),
                    }
                )

        return predictions

    active_model = model_bundle.get("model")

    if active_model is not None:
        probability = predict_probability_for_model(
            model=active_model,
            features=features,
            feature_names=feature_names,
        )

        predictions.append(
            build_model_prediction_item(
                model_name=active_model_name,
                probability=probability,
                metrics=model_bundle.get("metrics", {}),
                active_model_name=active_model_name,
            )
        )

    return predictions


def build_model_consensus(candidate_predictions: list[dict]) -> dict:
    """
    Računa konsenzus više modela.

    Ako postoji neriješen rezultat, sistem bira konzervativniju opciju,
    odnosno viši nivo rizika.
    """
    valid_predictions = [
        item for item in candidate_predictions
        if item.get("risk_level") in ["low", "medium", "high"]
    ]

    if not valid_predictions:
        return {
            "available": False,
            "message": "Konsenzus modela nije dostupan.",
            "total_models": 0,
            "risk_level": "unknown",
            "risk_label": "Nedostupno",
            "agreement_count": 0,
            "agreement_percent": "0.00%",
            "distribution": {},
        }

    distribution = {}

    for item in valid_predictions:
        risk_level = item.get("risk_level")
        distribution[risk_level] = distribution.get(risk_level, 0) + 1

    risk_priority = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    consensus_level = max(
        distribution.keys(),
        key=lambda level: (distribution[level], risk_priority.get(level, 0)),
    )

    total_models = len(valid_predictions)
    agreement_count = distribution[consensus_level]
    agreement_ratio = agreement_count / total_models

    label_map = {
        "low": "Nizak rizik",
        "medium": "Srednji rizik",
        "high": "Visok rizik",
    }

    class_map = {
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
    }

    return {
        "available": True,
        "method": "majority_vote_with_conservative_tie_break",
        "total_models": total_models,
        "risk_level": consensus_level,
        "risk_class": class_map.get(consensus_level, "Unknown"),
        "risk_label": label_map.get(consensus_level, "Nepoznato"),
        "agreement_count": agreement_count,
        "agreement_ratio": round(agreement_ratio, 4),
        "agreement_percent": f"{agreement_ratio * 100:.2f}%",
        "distribution": distribution,
        "message": (
            f"Konsenzus modela: {agreement_count}/{total_models} modela "
            f"procjenjuje {label_map.get(consensus_level, 'nepoznat rizik').lower()}."
        ),
    }


def predict_with_real_model(features: dict, model_bundle: dict) -> dict:
    """
    Pokreće pravi istrenirani ML model i vraća aktivnu predikciju,
    predikcije svih kandidata i konsenzus modela.
    """
    feature_names = model_bundle.get("feature_names", ML_FEATURE_NAMES)
    active_model_name = model_bundle.get("model_name", "Kapitalac ML Model")
    active_model = model_bundle.get("model")

    candidate_predictions = build_candidate_predictions(
        model_bundle=model_bundle,
        features=features,
    )

    active_prediction = next(
        (
            item for item in candidate_predictions
            if item.get("model_name") == active_model_name and item.get("risk_probability") is not None
        ),
        None,
    )

    if active_prediction is None and active_model is not None:
        probability = predict_probability_for_model(
            model=active_model,
            features=features,
            feature_names=feature_names,
        )

        active_prediction = build_model_prediction_item(
            model_name=active_model_name,
            probability=probability,
            metrics=model_bundle.get("metrics", {}),
            active_model_name=active_model_name,
        )

    if active_prediction is None:
        probability = 0.50
        classification = classify_ml_probability(probability)
    else:
        probability = active_prediction.get("risk_probability", 0.50)
        classification = active_prediction.get(
            "classification",
            classify_ml_probability(probability),
        )

    model_consensus = build_model_consensus(candidate_predictions)

    return {
        "model_name": active_model_name,
        "model_type": model_bundle.get("model_type", "machine_learning"),
        "is_ml_model_loaded": True,
        "risk_probability": probability,
        "risk_probability_percent": f"{probability * 100:.2f}%",
        "classification": classification,
        "features": features,
        "candidate_predictions": candidate_predictions,
        "model_consensus": model_consensus,
        "model_metrics": model_bundle.get("metrics", {}),
        "all_model_metrics": model_bundle.get("all_model_metrics", {}),
        "training_info": model_bundle.get("training_info", {}),
        "candidate_model_names": model_bundle.get("candidate_model_names", []),
        "hybrid_note": model_bundle.get("hybrid_note"),
        "note": (
            "Aktivna procjena dolazi iz modela sa najboljim metrikama, "
            "dok se procjene ostalih kandidata prikazuju radi transparentnosti."
        ),
    }


def predict_ml_risk(scoring: dict, validation: dict) -> dict:
    """
    Glavna funkcija za ML procjenu rizika.
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
        "candidate_predictions": [],
        "model_consensus": {
            "available": False,
            "message": "Konsenzus modela nije dostupan jer ML model nije učitan.",
        },
        "note": (
            "Ovo je privremeni rule-based rezultat. "
            "Kada se istrenira pravi model i sačuva u models/kapitalac_ml_model.joblib, "
            "API će automatski koristiti ML model."
        ),
    }