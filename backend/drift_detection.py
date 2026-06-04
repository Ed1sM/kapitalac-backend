from backend.ml_model import ML_FEATURE_NAMES, load_ml_model


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


def get_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def detect_feature_drift(features: dict) -> dict:
    model_bundle = load_ml_model()

    if model_bundle is None:
        return {
            "is_available": False,
            "is_drift_detected": False,
            "method": "reference_stats",
            "message": "Drift detection nije dostupan jer ML model nije učitan.",
            "drifted_features": [],
        }

    reference_stats = model_bundle.get("reference_stats", {})

    if not reference_stats:
        return {
            "is_available": False,
            "is_drift_detected": False,
            "method": "reference_stats",
            "message": "Referentna statistika trening skupa nije pronađena u model bundle-u.",
            "drifted_features": [],
        }

    drifted_features = []

    for feature_name in ML_FEATURE_NAMES:
        if feature_name not in reference_stats:
            continue

        value = features.get(feature_name)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        stats = reference_stats[feature_name]

        mean = stats.get("mean")
        std = stats.get("std")
        q05 = stats.get("q05")
        q95 = stats.get("q95")
        min_value = stats.get("min")
        max_value = stats.get("max")

        severity = None
        reason = None
        z_score = None

        if std is not None and std > 0 and mean is not None:
            z_score = abs((value - mean) / std)

            if z_score >= 4:
                severity = "high"
                reason = "Vrijednost je ekstremno udaljena od trening distribucije."
            elif z_score >= 3:
                severity = "medium"
                reason = "Vrijednost je značajno udaljena od trening distribucije."

        if severity is None and q05 is not None and q95 is not None:
            if value < q05 or value > q95:
                severity = "low"
                reason = "Vrijednost je izvan 5–95% raspona trening distribucije."

        if severity is not None:
            drifted_features.append(
                {
                    "feature": feature_name,
                    "label": get_label(feature_name),
                    "value": round(value, 6),
                    "severity": severity,
                    "reason": reason,
                    "z_score": round(z_score, 4) if z_score is not None else None,
                    "reference": {
                        "mean": mean,
                        "std": std,
                        "min": min_value,
                        "max": max_value,
                        "q05": q05,
                        "q95": q95,
                    },
                }
            )

    severity_order = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    drifted_features = sorted(
        drifted_features,
        key=lambda item: severity_order.get(item["severity"], 0),
        reverse=True,
    )

    high_count = sum(1 for item in drifted_features if item["severity"] == "high")
    medium_count = sum(1 for item in drifted_features if item["severity"] == "medium")
    low_count = sum(1 for item in drifted_features if item["severity"] == "low")

    is_drift_detected = high_count > 0 or medium_count > 0

    if is_drift_detected:
        message = "Detektovano je odstupanje novih podataka od trening distribucije."
    elif drifted_features:
        message = "Postoje manja odstupanja, ali bez ozbiljnog drift signala."
    else:
        message = "Nijesu detektovana značajna odstupanja od trening distribucije."

    return {
        "is_available": True,
        "is_drift_detected": is_drift_detected,
        "method": "single_observation_reference_stats",
        "message": message,
        "summary": {
            "total_drifted_features": len(drifted_features),
            "high_severity_count": high_count,
            "medium_severity_count": medium_count,
            "low_severity_count": low_count,
        },
        "drifted_features": drifted_features,
    }