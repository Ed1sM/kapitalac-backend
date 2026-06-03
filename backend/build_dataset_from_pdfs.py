from pathlib import Path

import pandas as pd

from backend.altman_score import calculate_altman_scores
from backend.ml_model import ML_FEATURE_NAMES, prepare_ml_features
from backend.pdf_extractor import extract_financials_from_pdf
from backend.validation import validate_financial_data


RAW_DATA_FOLDER = Path("data/raw")
OUTPUT_PATH = Path("data/processed/company_features_from_pdfs.csv")


def process_pdf(pdf_path: Path) -> dict:
    financials = extract_financials_from_pdf(str(pdf_path))
    financials["file_name"] = pdf_path.name

    validation = validate_financial_data(financials)
    scoring = calculate_altman_scores(financials)
    features = prepare_ml_features(scoring)

    altman_private = scoring.get("altman_private", {})
    altman_original = scoring.get("altman_original", {})

    private_classification = altman_private.get("classification", {})
    original_classification = altman_original.get("classification", {})

    row = {
        "file_name": financials.get("file_name"),
        "company_name": financials.get("company_name"),
        "registration_number": financials.get("registration_number"),
        "activity_code": financials.get("activity_code"),
        "report_year": financials.get("report_year"),

        # Ovu kolonu ručno popunjavamo kasnije.
        # 0 = stabilna firma
        # 1 = bankrot / stečaj / ozbiljan finansijski problem
        "target_bankrupt": "",

        "altman_z_score": altman_original.get("score"),
        "altman_z_zone": original_classification.get("zone"),
        "altman_z_prime_score": altman_private.get("score"),
        "altman_z_prime_zone": private_classification.get("zone"),

        "data_quality_score": validation.get("quality", {}).get("score"),
        "data_quality_label": validation.get("quality", {}).get("label"),
        "warnings_count": validation.get("quality", {}).get("warnings_count"),

        "total_assets": financials.get("total_assets"),
        "equity": financials.get("equity"),
        "total_liabilities": scoring.get("variables", {}).get("total_liabilities"),
        "current_assets": financials.get("current_assets"),
        "short_term_liabilities": financials.get("short_term_liabilities"),
        "sales_revenue": financials.get("sales_revenue"),
        "operating_profit": financials.get("operating_profit"),
        "net_profit": financials.get("net_profit"),
    }

    for feature_name in ML_FEATURE_NAMES:
        row[feature_name] = features.get(feature_name)

    return row


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DATA_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"Nema PDF fajlova u folderu: {RAW_DATA_FOLDER}")
        print("Dodaj finansijske izvještaje u data/raw folder.")
        return

    rows = []

    print("Pronađeni PDF fajlovi:")

    for pdf_path in pdf_files:
        print(f"- {pdf_path.name}")

        try:
            row = process_pdf(pdf_path)
            rows.append(row)

        except Exception as error:
            print(f"GREŠKA za fajl {pdf_path.name}: {error}")

    if not rows:
        print("Nijedan PDF nije uspješno obrađen.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print(f"Dataset iz PDF-ova je kreiran: {OUTPUT_PATH}")
    print()
    print("Pregled:")
    preview_columns = [
        "file_name",
        "company_name",
        "report_year",
        "target_bankrupt",
        "altman_z_prime_score",
        "altman_z_prime_zone",
        "data_quality_score",
    ]

    existing_columns = [column for column in preview_columns if column in df.columns]
    print(df[existing_columns].to_string(index=False))


if __name__ == "__main__":
    main()