from typing import Any

import numpy as np
import pandas as pd

from backend.ml_model import ML_FEATURE_NAMES, load_ml_model


try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


FEATURE_LABELS = {
    "current_ratio": "Tekuća likvidnost",
    "quick_ratio": "Brza likvidnost",
    "cash_ratio": "Cash ratio",
    "debt_to_assets": "Zaduženost u odnosu na aktivu",
    "debt_to_equity": "Zaduženost u odnosu na kapital",
    "roa": "ROA",
    "operating_margin": "Operativna marža",
    "net_profit_margin": "Neto profitna marža",
    "asset_turnover": "Obrt aktive",
    "working_capital_to_assets": "Radni kapital / aktiva",
    "retained_earnings_to_assets": "Zadržana dobit / aktiva",
    "ebit_to_assets": "EBIT / aktiva",
    "equity_to_liabilities": "Kapital / obaveze",
    "sales_to_assets": "Prihodi / aktiva",
}


def get_feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def make_factor_explanation(feature_name: str, value: float, impact: float) -> str:
    label = get_feature_label(feature_name)

    if impact > 0:
        direction = "povećava procijenjeni rizik"
    elif impact < 0:
        direction = "smanjuje procijenjeni rizik"
    else:
        direction = "nema značajan uticaj na rizik"

    return f"{label} ({round(value, 4)}) {direction}."


def format_factors(features: dict, impacts: dict) -> dict:
    all_factors = []

    for feature_name, impact in impacts.items():
        value = features.get(feature_name, 0.0)

        factor = {
            "feature": feature_name,
            "label": get_feature_label(feature_name),
            "feature_value": round(float(value), 4),
            "impact_value": round(float(impact), 6),
            "absolute_impact": round(abs(float(impact)), 6),
            "impact_direction": (
                "increases_risk"
                if impact > 0
                else "decreases_risk"
                if impact < 0
                else "neutral"
            ),
            "explanation": make_factor_explanation(
                feature_name=feature_name,
                value=float(value),
                impact=float(impact),
            ),
        }

        all_factors.append(factor)

    all_factors = sorted(
        all_factors,
        key=lambda item: item["absolute_impact"],
        reverse=True,
    )

    top_positive = [
        factor for factor in all_factors
        if factor["impact_direction"] == "increases_risk"
    ][:5]

    top_negative = [
        factor for factor in all_factors
        if factor["impact_direction"] == "decreases_risk"
    ][:5]

    return {
        "top_factors": all_factors[:8],
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "all_factors": all_factors,
    }


def get_pipeline_classifier(model: Any):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("classifier")

    return model


def transform_pipeline_input(model: Any, row: pd.DataFrame):
    if hasattr(model, "steps") and len(model.steps) > 1:
        preprocessing_pipeline = model[:-1]
        return preprocessing_pipeline.transform(row)

    return row.values


def explain_with_shap(model_bundle: dict, features: dict) -> dict | None:
    if not SHAP_AVAILABLE:
        return None

    model = model_bundle.get("model")
    model_name = model_bundle.get("model_name")
    feature_names = model_bundle.get("feature_names", ML_FEATURE_NAMES)

    if model is None:
        return None

    classifier = get_pipeline_classifier(model)

    if model_name != "XGBoost":
        return None

    row = pd.DataFrame(
        [[features.get(feature_name, 0.0) for feature_name in feature_names]],
        columns=feature_names,
    )

    transformed_row = transform_pipeline_input(model, row)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    impacts = {
        feature_name: float(shap_values[index])
        for index, feature_name in enumerate(feature_names)
    }

    factors = format_factors(features=features, impacts=impacts)

    return {
        "method": "SHAP TreeExplainer",
        "is_shap_available": True,
        "model_name": model_name,
        "description": (
            "SHAP vrijednosti pokazuju koliko je svaki finansijski pokazatelj "
            "uticao na povećanje ili smanjenje procijenjenog rizika."
        ),
        **factors,
    }


def explain_with_logistic_coefficients(model_bundle: dict, features: dict) -> dict | None:
    model = model_bundle.get("model")
    feature_names = model_bundle.get("feature_names", ML_FEATURE_NAMES)

    if model is None or not hasattr(model, "named_steps"):
        return None

    classifier = model.named_steps.get("classifier")

    if classifier is None or not hasattr(classifier, "coef_"):
        return None

    row = pd.DataFrame(
        [[features.get(feature_name, 0.0) for feature_name in feature_names]],
        columns=feature_names,
    )

    transformed_row = transform_pipeline_input(model, row)
    coefficients = classifier.coef_[0]

    impacts = {
        feature_name: float(coefficients[index] * transformed_row[0][index])
        for index, feature_name in enumerate(feature_names)
    }

    factors = format_factors(features=features, impacts=impacts)

    return {
        "method": "Logistic Regression coefficient contribution",
        "is_shap_available": False,
        "model_name": model_bundle.get("model_name"),
        "description": (
            "Model nije XGBoost, pa je prikazano objašnjenje na osnovu doprinosa "
            "koeficijenata logističke regresije."
        ),
        **factors,
    }


def explain_with_feature_importance(model_bundle: dict, features: dict) -> dict | None:
    model = model_bundle.get("model")
    feature_names = model_bundle.get("feature_names", ML_FEATURE_NAMES)

    if model is None:
        return None

    classifier = get_pipeline_classifier(model)

    if not hasattr(classifier, "feature_importances_"):
        return None

    importances = classifier.feature_importances_

    impacts = {
        feature_name: float(importances[index])
        for index, feature_name in enumerate(feature_names)
    }

    factors = format_factors(features=features, impacts=impacts)

    return {
        "method": "Feature importance",
        "is_shap_available": False,
        "model_name": model_bundle.get("model_name"),
        "description": (
            "Prikazane su najvažnije varijable modela. "
            "Ovo je fallback objašnjenje ako SHAP nije dostupan."
        ),
        **factors,
    }


def build_prediction_explanation(ml_prediction: dict) -> dict:
    features = ml_prediction.get("features", {})
    model_bundle = load_ml_model()

    if model_bundle is None:
        return {
            "method": "rule_based_explanation",
            "is_shap_available": False,
            "description": "ML model nije učitan, koristi se osnovno pravilo za objašnjenje.",
            "top_factors": [],
            "top_positive_factors": [],
            "top_negative_factors": [],
            "all_factors": [],
        }

    try:
        shap_explanation = explain_with_shap(model_bundle, features)

        if shap_explanation is not None:
            return shap_explanation
    except Exception as error:
        shap_error = str(error)
    else:
        shap_error = None

    logistic_explanation = explain_with_logistic_coefficients(model_bundle, features)

    if logistic_explanation is not None:
        logistic_explanation["shap_error"] = shap_error
        return logistic_explanation

    importance_explanation = explain_with_feature_importance(model_bundle, features)

    if importance_explanation is not None:
        importance_explanation["shap_error"] = shap_error
        return importance_explanation

    return {
        "method": "unavailable",
        "is_shap_available": False,
        "description": "Objašnjenje modela nije dostupno za trenutno učitani model.",
        "shap_error": shap_error,
        "top_factors": [],
        "top_positive_factors": [],
        "top_negative_factors": [],
        "all_factors": [],
    }