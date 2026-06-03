from typing import Optional


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """
    Sigurno dijeljenje.
    Ako podatak ne postoji ili je imenilac 0, vraća None.
    """
    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def round_value(value: Optional[float], decimals: int = 4) -> Optional[float]:
    """
    Zaokružuje broj ako postoji.
    """
    if value is None:
        return None

    return round(value, decimals)


def get_total_liabilities(financials: dict) -> Optional[float]:
    """
    Računa ukupne obaveze.

    Najbolji pristup:
    ukupne obaveze = ukupna aktiva - kapital
    """
    total_assets = financials.get("total_assets")
    equity = financials.get("equity")

    if total_assets is not None and equity is not None:
        return total_assets - equity

    long_term_liabilities = financials.get("long_term_liabilities") or 0
    short_term_liabilities = financials.get("short_term_liabilities") or 0

    return long_term_liabilities + short_term_liabilities


def classify_altman_original(z_score: Optional[float]) -> dict:
    """
    Klasifikacija za originalni Altman Z-Score.

    Granice:
    Z > 2.99       - Safe zone
    1.81 - 2.99    - Grey zone
    Z < 1.81       - Distress zone
    """
    if z_score is None:
        return {
            "zone": "Unknown",
            "label": "Nije moguće izračunati",
            "risk_level": "unknown",
        }

    if z_score > 2.99:
        return {
            "zone": "Safe zone",
            "label": "Stabilno poslovanje",
            "risk_level": "low",
        }

    if z_score >= 1.81:
        return {
            "zone": "Grey zone",
            "label": "Siva zona",
            "risk_level": "medium",
        }

    return {
        "zone": "Distress zone",
        "label": "Visok rizik bankrota",
        "risk_level": "high",
    }


def classify_altman_private(z_prime_score: Optional[float]) -> dict:
    """
    Klasifikacija za Altman Z'-Score za privatne firme.

    Granice:
    Z' > 2.90       - Safe zone
    1.21 - 2.90     - Grey zone
    Z' < 1.21       - Distress zone
    """
    if z_prime_score is None:
        return {
            "zone": "Unknown",
            "label": "Nije moguće izračunati",
            "risk_level": "unknown",
        }

    if z_prime_score > 2.90:
        return {
            "zone": "Safe zone",
            "label": "Stabilno poslovanje",
            "risk_level": "low",
        }

    if z_prime_score >= 1.21:
        return {
            "zone": "Grey zone",
            "label": "Siva zona",
            "risk_level": "medium",
        }

    return {
        "zone": "Distress zone",
        "label": "Visok rizik bankrota",
        "risk_level": "high",
    }


def calculate_altman_variables(financials: dict) -> dict:
    """
    Računa X1-X5 varijable za Altman model.

    X1 = Radni kapital / Ukupna aktiva
    X2 = Neraspoređena dobit / Ukupna aktiva
    X3 = Poslovni rezultat / Ukupna aktiva
    X4 = Kapital / Ukupne obaveze
    X5 = Prihodi od prodaje / Ukupna aktiva
    """
    total_assets = financials.get("total_assets")
    current_assets = financials.get("current_assets")
    short_term_liabilities = financials.get("short_term_liabilities")
    retained_earnings = financials.get("retained_earnings")
    operating_profit = financials.get("operating_profit")
    equity = financials.get("equity")
    sales_revenue = financials.get("sales_revenue")

    total_liabilities = get_total_liabilities(financials)

    working_capital = None
    if current_assets is not None and short_term_liabilities is not None:
        working_capital = current_assets - short_term_liabilities

    x1 = safe_divide(working_capital, total_assets)
    x2 = safe_divide(retained_earnings, total_assets)
    x3 = safe_divide(operating_profit, total_assets)
    x4 = safe_divide(equity, total_liabilities)
    x5 = safe_divide(sales_revenue, total_assets)

    return {
        "working_capital": working_capital,
        "total_liabilities": total_liabilities,
        "x1_working_capital_to_assets": x1,
        "x2_retained_earnings_to_assets": x2,
        "x3_ebit_to_assets": x3,
        "x4_equity_to_liabilities": x4,
        "x5_sales_to_assets": x5,
    }


def calculate_financial_ratios(financials: dict) -> dict:
    """
    Računa dodatne finansijske pokazatelje za dashboard i budući ML model.
    """
    total_assets = financials.get("total_assets")
    current_assets = financials.get("current_assets")
    inventory = financials.get("inventory")
    cash = financials.get("cash")
    short_term_liabilities = financials.get("short_term_liabilities")
    equity = financials.get("equity")
    sales_revenue = financials.get("sales_revenue")
    operating_profit = financials.get("operating_profit")
    net_profit = financials.get("net_profit")
    average_employees = financials.get("average_employees")

    total_liabilities = get_total_liabilities(financials)

    quick_assets = None
    if current_assets is not None:
        quick_assets = current_assets - (inventory or 0)

    return {
        "current_ratio": safe_divide(current_assets, short_term_liabilities),
        "quick_ratio": safe_divide(quick_assets, short_term_liabilities),
        "cash_ratio": safe_divide(cash, short_term_liabilities),
        "debt_to_assets": safe_divide(total_liabilities, total_assets),
        "debt_to_equity": safe_divide(total_liabilities, equity),
        "roa": safe_divide(net_profit, total_assets),
        "operating_margin": safe_divide(operating_profit, sales_revenue),
        "net_profit_margin": safe_divide(net_profit, sales_revenue),
        "asset_turnover": safe_divide(sales_revenue, total_assets),
        "revenue_per_employee": safe_divide(sales_revenue, average_employees),
    }


def build_explanation(
    financials: dict,
    variables: dict,
    ratios: dict,
    private_classification: dict,
) -> list[str]:
    """
    Pravi jednostavno poslovno objašnjenje rezultata.
    Kasnije ćemo ovo proširiti ML i SHAP objašnjenjima.
    """
    explanation = []

    company_name = financials.get("company_name") or "Kompanija"
    risk_label = private_classification.get("label", "Nije moguće izračunati")

    explanation.append(
        f"{company_name} je prema Altman Z'-Score modelu klasifikovana kao: {risk_label}."
    )

    current_ratio = ratios.get("current_ratio")
    debt_to_assets = ratios.get("debt_to_assets")
    roa = ratios.get("roa")
    net_profit_margin = ratios.get("net_profit_margin")

    x1 = variables.get("x1_working_capital_to_assets")
    x3 = variables.get("x3_ebit_to_assets")
    x4 = variables.get("x4_equity_to_liabilities")

    if current_ratio is not None:
        if current_ratio >= 2:
            explanation.append(
                "Likvidnost je jaka jer su obrtna sredstva najmanje dvostruko veća od kratkoročnih obaveza."
            )
        elif current_ratio >= 1:
            explanation.append(
                "Likvidnost je prihvatljiva jer su obrtna sredstva veća od kratkoročnih obaveza."
            )
        else:
            explanation.append(
                "Likvidnost je slaba jer kratkoročne obaveze premašuju obrtna sredstva."
            )

    if debt_to_assets is not None:
        if debt_to_assets < 0.4:
            explanation.append("Zaduženost je niska u odnosu na ukupnu aktivu.")
        elif debt_to_assets < 0.7:
            explanation.append("Zaduženost je umjerena i zahtijeva praćenje.")
        else:
            explanation.append("Zaduženost je visoka i može predstavljati značajan rizik.")

    if roa is not None:
        if roa > 0.05:
            explanation.append(
                "Profitabilnost imovine je dobra jer kompanija ostvaruje pozitivan prinos na ukupnu aktivu."
            )
        elif roa > 0:
            explanation.append(
                "Kompanija ostvaruje pozitivan, ali nizak prinos na aktivu."
            )
        else:
            explanation.append(
                "Kompanija ima negativan prinos na aktivu, što je signal povećanog rizika."
            )

    if net_profit_margin is not None:
        if net_profit_margin > 0.1:
            explanation.append("Neto profitna marža je snažna.")
        elif net_profit_margin > 0:
            explanation.append("Neto profitna marža je pozitivna, ali relativno skromna.")
        else:
            explanation.append("Neto profitna marža je negativna.")

    if x1 is not None and x1 < 0:
        explanation.append("Radni kapital je negativan, što nepovoljno utiče na Altman rezultat.")

    if x3 is not None and x3 > 0:
        explanation.append("Pozitivan poslovni rezultat doprinosi boljem Altman rezultatu.")

    if x4 is not None and x4 > 1:
        explanation.append(
            "Kapital je veći od ukupnih obaveza, što značajno smanjuje finansijski rizik."
        )

    return explanation


def calculate_altman_scores(financials: dict) -> dict:
    """
    Računa:
    - originalni Altman Z-Score
    - Altman Z'-Score za privatne firme
    - dodatne finansijske pokazatelje
    - tekstualno objašnjenje rezultata
    """
    variables = calculate_altman_variables(financials)

    x1 = variables["x1_working_capital_to_assets"]
    x2 = variables["x2_retained_earnings_to_assets"]
    x3 = variables["x3_ebit_to_assets"]
    x4 = variables["x4_equity_to_liabilities"]
    x5 = variables["x5_sales_to_assets"]

    required_values = [x1, x2, x3, x4, x5]

    if any(value is None for value in required_values):
        z_score = None
        z_prime_score = None
    else:
        z_score = (
            (1.2 * x1)
            + (1.4 * x2)
            + (3.3 * x3)
            + (0.6 * x4)
            + (0.999 * x5)
        )

        z_prime_score = (
            (0.717 * x1)
            + (0.847 * x2)
            + (3.107 * x3)
            + (0.420 * x4)
            + (0.998 * x5)
        )

    ratios = calculate_financial_ratios(financials)

    rounded_variables = {
        key: round_value(value)
        for key, value in variables.items()
    }

    rounded_ratios = {
        key: round_value(value)
        for key, value in ratios.items()
    }

    original_classification = classify_altman_original(z_score)
    private_classification = classify_altman_private(z_prime_score)

    explanation = build_explanation(
        financials=financials,
        variables=variables,
        ratios=ratios,
        private_classification=private_classification,
    )

    return {
        "altman_original": {
            "score": round_value(z_score),
            "classification": original_classification,
            "note": (
                "Originalni Altman Z-Score koristi tržišnu vrijednost kapitala. "
                "Za firme koje nijesu na berzi ovdje se koristi knjigovodstveni kapital kao praktična aproksimacija."
            ),
        },
        "altman_private": {
            "score": round_value(z_prime_score),
            "classification": private_classification,
            "note": (
                "Altman Z'-Score je pogodniji za privatne firme jer koristi knjigovodstvenu vrijednost kapitala."
            ),
        },
        "variables": rounded_variables,
        "ratios": rounded_ratios,
        "explanation": explanation,
    }