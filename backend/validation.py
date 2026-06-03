from typing import Optional


REQUIRED_FIELDS_FOR_ALTMAN = [
    "total_assets",
    "current_assets",
    "short_term_liabilities",
    "retained_earnings",
    "operating_profit",
    "equity",
    "sales_revenue",
]


IMPORTANT_FIELDS_FOR_DASHBOARD = [
    "inventory",
    "cash",
    "short_term_receivables",
    "long_term_liabilities",
    "net_profit",
    "operating_cash_flow",
    "average_employees",
]


def is_missing(value) -> bool:
    return value is None or value == ""


def validate_required_fields(financials: dict) -> list[dict]:
    warnings = []

    for field in REQUIRED_FIELDS_FOR_ALTMAN:
        if is_missing(financials.get(field)):
            warnings.append(
                {
                    "type": "missing_required_field",
                    "field": field,
                    "severity": "high",
                    "message": f"Nedostaje obavezni podatak za Altman model: {field}.",
                }
            )

    return warnings


def validate_dashboard_fields(financials: dict) -> list[dict]:
    warnings = []

    for field in IMPORTANT_FIELDS_FOR_DASHBOARD:
        if is_missing(financials.get(field)):
            warnings.append(
                {
                    "type": "missing_dashboard_field",
                    "field": field,
                    "severity": "medium",
                    "message": f"Nedostaje podatak za dodatnu analizu/dashboard: {field}.",
                }
            )

    return warnings


def validate_balance_sheet(financials: dict) -> list[dict]:
    warnings = []

    total_assets = financials.get("total_assets")
    total_liabilities_and_equity = financials.get("total_liabilities_and_equity")
    equity = financials.get("equity")

    if total_assets is not None and total_liabilities_and_equity is not None:
        difference = abs(total_assets - total_liabilities_and_equity)

        if difference > 1:
            warnings.append(
                {
                    "type": "balance_sheet_mismatch",
                    "field": "total_assets",
                    "severity": "high",
                    "message": (
                        "Ukupna aktiva nije jednaka ukupnoj pasivi. "
                        f"Razlika iznosi {difference}."
                    ),
                }
            )

    if total_assets is not None and total_assets <= 0:
        warnings.append(
            {
                "type": "invalid_total_assets",
                "field": "total_assets",
                "severity": "high",
                "message": "Ukupna aktiva mora biti veća od nule.",
            }
        )

    if equity is not None and equity < 0:
        warnings.append(
            {
                "type": "negative_equity",
                "field": "equity",
                "severity": "high",
                "message": "Kapital je negativan, što ukazuje na povećan finansijski rizik.",
            }
        )

    return warnings


def validate_extraction_sources(financials: dict) -> list[dict]:
    warnings = []

    extraction_details = financials.get("extraction_details", {})

    for field in REQUIRED_FIELDS_FOR_ALTMAN:
        details = extraction_details.get(field, {})
        source_line = details.get("source_line")

        if not source_line:
            warnings.append(
                {
                    "type": "missing_source_line",
                    "field": field,
                    "severity": "medium",
                    "message": f"Nije pronađen izvorni red u PDF-u za polje: {field}.",
                }
            )

    return warnings


def calculate_data_quality_score(warnings: list[dict]) -> dict:
    score = 100

    for warning in warnings:
        severity = warning.get("severity")

        if severity == "high":
            score -= 20
        elif severity == "medium":
            score -= 8
        else:
            score -= 3

    score = max(score, 0)

    if score >= 90:
        label = "Visok kvalitet ekstrakcije"
    elif score >= 70:
        label = "Dobar kvalitet ekstrakcije"
    elif score >= 50:
        label = "Srednji kvalitet ekstrakcije"
    else:
        label = "Nizak kvalitet ekstrakcije"

    return {
        "score": score,
        "label": label,
        "warnings_count": len(warnings),
    }


def validate_financial_data(financials: dict) -> dict:
    warnings = []

    warnings.extend(validate_required_fields(financials))
    warnings.extend(validate_dashboard_fields(financials))
    warnings.extend(validate_balance_sheet(financials))
    warnings.extend(validate_extraction_sources(financials))

    quality = calculate_data_quality_score(warnings)

    is_valid_for_altman = not any(
        warning.get("severity") == "high"
        and warning.get("type") in ["missing_required_field", "invalid_total_assets"]
        for warning in warnings
    )

    return {
        "quality": quality,
        "warnings": warnings,
        "is_valid_for_altman": is_valid_for_altman,
    }


def format_number(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None

    try:
        return f"{value:,.0f}".replace(",", ".")
    except (ValueError, TypeError):
        return str(value)


def format_percent(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None

    try:
        return f"{value * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def build_lovable_payload(
    financials: dict,
    scoring: dict,
    validation: dict,
    ml_prediction: dict | None = None,
) -> dict:
    private_score = scoring.get("altman_private", {})
    original_score = scoring.get("altman_original", {})
    ratios = scoring.get("ratios", {})
    variables = scoring.get("variables", {})

    private_classification = private_score.get("classification", {})
    original_classification = original_score.get("classification", {})

    return {
        "brand": {
            "name": "Kapitalac",
            "tagline": "Pametna procjena finansijskog zdravlja kompanija.",
        },
        "company_card": {
            "company_name": financials.get("company_name"),
            "registration_number": financials.get("registration_number"),
            "activity_code": financials.get("activity_code"),
            "report_year": financials.get("report_year"),
            "file_name": financials.get("file_name"),
        },
        "risk_summary": {
            "primary_score_name": "Altman Z'-Score",
            "primary_score": private_score.get("score"),
            "primary_zone": private_classification.get("zone"),
            "primary_label": private_classification.get("label"),
            "risk_level": private_classification.get("risk_level"),
            "secondary_score_name": "Altman Z-Score",
            "secondary_score": original_score.get("score"),
            "secondary_zone": original_classification.get("zone"),
        },
        "ml_prediction": ml_prediction,
        "financial_highlights": {
            "total_assets": financials.get("total_assets"),
            "total_assets_formatted": format_number(financials.get("total_assets")),
            "equity": financials.get("equity"),
            "equity_formatted": format_number(financials.get("equity")),
            "total_liabilities": variables.get("total_liabilities"),
            "total_liabilities_formatted": format_number(variables.get("total_liabilities")),
            "sales_revenue": financials.get("sales_revenue"),
            "sales_revenue_formatted": format_number(financials.get("sales_revenue")),
            "net_profit": financials.get("net_profit"),
            "net_profit_formatted": format_number(financials.get("net_profit")),
            "working_capital": variables.get("working_capital"),
            "working_capital_formatted": format_number(variables.get("working_capital")),
        },
        "ratio_cards": {
            "current_ratio": ratios.get("current_ratio"),
            "quick_ratio": ratios.get("quick_ratio"),
            "debt_to_assets": ratios.get("debt_to_assets"),
            "debt_to_assets_percent": format_percent(ratios.get("debt_to_assets")),
            "roa": ratios.get("roa"),
            "roa_percent": format_percent(ratios.get("roa")),
            "net_profit_margin": ratios.get("net_profit_margin"),
            "net_profit_margin_percent": format_percent(ratios.get("net_profit_margin")),
            "asset_turnover": ratios.get("asset_turnover"),
        },
        "explanation": scoring.get("explanation", []),
        "data_quality": validation.get("quality"),
        "warnings": validation.get("warnings", []),
    }