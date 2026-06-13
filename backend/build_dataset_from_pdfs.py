from pathlib import Path

import pandas as pd

from backend.altman_score import calculate_altman_scores
from backend.pdf_extractor import extract_financials_from_pdf
from backend.validation import validate_financial_data


RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/company_features_from_pdfs.csv")


def build_row_from_pdf(pdf_path: Path) -> dict:
    extracted = extract_financials_from_pdf(str(pdf_path))
    scoring = calculate_altman_scores(extracted)
    validation = validate_financial_data(extracted)

    variables = scoring.get("variables", {})
    ratios = scoring.get("ratios", {})
    altman_private = scoring.get("altman_private", {})
    altman_original = scoring.get("altman_original", {})

    private_classification = altman_private.get("classification", {})
    original_classification = altman_original.get("classification", {})

    row = {
        "file_name": pdf_path.name,
        "company_name": extracted.get("company_name"),
        "registration_number": extracted.get("registration_number"),
        "activity_code": extracted.get("activity_code"),
        "report_year": extracted.get("report_year"),
        "data_quality_score": validation.get("quality", {}).get("score"),
        "data_quality_label": validation.get("quality", {}).get("label"),
        "warnings_count": validation.get("quality", {}).get("warnings_count"),
        "is_valid_for_altman": validation.get("is_valid_for_altman"),
        "altman_z_score": altman_original.get("score"),
        "altman_z_zone": original_classification.get("zone"),
        "altman_z_risk_level": original_classification.get("risk_level"),
        "altman_z_prime_score": altman_private.get("score"),
        "altman_z_prime_zone": private_classification.get("zone"),
        "altman_z_prime_risk_level": private_classification.get("risk_level"),
        "working_capital": variables.get("working_capital"),
        "total_liabilities": variables.get("total_liabilities"),
        "working_capital_to_assets": variables.get("x1_working_capital_to_assets"),
        "retained_earnings_to_assets": variables.get("x2_retained_earnings_to_assets"),
        "ebit_to_assets": variables.get("x3_ebit_to_assets"),
        "equity_to_liabilities": variables.get("x4_equity_to_liabilities"),
        "sales_to_assets": variables.get("x5_sales_to_assets"),
        "current_ratio": ratios.get("current_ratio"),
        "quick_ratio": ratios.get("quick_ratio"),
        "cash_ratio": ratios.get("cash_ratio"),
        "debt_to_assets": ratios.get("debt_to_assets"),
        "debt_to_equity": ratios.get("debt_to_equity"),
        "roa": ratios.get("roa"),
        "operating_margin": ratios.get("operating_margin"),
        "net_profit_margin": ratios.get("net_profit_margin"),
        "asset_turnover": ratios.get("asset_turnover"),
        "revenue_per_employee": ratios.get("revenue_per_employee"),
    }

    for field in [
        "total_assets",
        "fixed_assets",
        "current_assets",
        "inventory",
        "short_term_receivables",
        "cash",
        "equity",
        "retained_earnings",
        "long_term_liabilities",
        "short_term_liabilities",
        "sales_revenue",
        "operating_profit",
        "financial_result",
        "profit_before_tax",
        "net_profit",
        "operating_cash_flow",
        "average_employees",
    ]:
        row[field] = extracted.get(field)

    return row


def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Nije pronađen folder: {RAW_DATA_DIR}. "
            "Kreiraj folder data/raw i u njega dodaj PDF izvještaje."
        )

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nema PDF fajlova u folderu: {RAW_DATA_DIR}."
        )

    rows = []

    for pdf_path in pdf_files:
        print(f"Obrađujem PDF: {pdf_path.name}")

        try:
            row = build_row_from_pdf(pdf_path)
            rows.append(row)
            print("Uspješno obrađeno.")
        except Exception as error:
            print(f"Greška pri obradi fajla {pdf_path.name}: {error}")

    if not rows:
        raise RuntimeError("Nijedan PDF nije uspješno obrađen.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("Dataset je uspješno kreiran.")
    print(f"Broj obrađenih kompanija: {len(df)}")
    print(f"Sačuvano u: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()