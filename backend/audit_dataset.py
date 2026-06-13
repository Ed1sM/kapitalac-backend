from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/company_features_from_pdfs.csv")
AUDIT_OUTPUT_PATH = Path("data/processed/dataset_audit_report.csv")
CLEAN_OUTPUT_PATH = Path("data/processed/company_features_clean_candidates.csv")


REQUIRED_COLUMNS = [
    "file_name",
    "company_name",
    "registration_number",
    "report_year",
    "data_quality_score",
    "is_valid_for_altman",
    "altman_z_prime_score",
    "altman_z_prime_zone",
    "total_assets",
    "current_assets",
    "short_term_liabilities",
    "retained_earnings",
    "operating_profit",
    "equity",
    "sales_revenue",
]


def is_missing(value) -> bool:
    return pd.isna(value) or value == ""


def audit_row(row: pd.Series) -> dict:
    issues = []
    severity = "low"

    for column in REQUIRED_COLUMNS:
        if column not in row.index or is_missing(row[column]):
            issues.append(f"Nedostaje kolona ili vrijednost: {column}")
            severity = "high"

    total_assets = row.get("total_assets")
    equity = row.get("equity")
    short_term_liabilities = row.get("short_term_liabilities")
    sales_revenue = row.get("sales_revenue")
    data_quality_score = row.get("data_quality_score")
    altman_zone = row.get("altman_z_prime_zone")

    if not is_missing(total_assets) and total_assets <= 0:
        issues.append("Ukupna aktiva je manja ili jednaka nuli.")
        severity = "high"

    if not is_missing(equity) and equity < 0:
        issues.append("Kapital je negativan.")
        severity = "medium" if severity != "high" else severity

    if not is_missing(short_term_liabilities) and short_term_liabilities < 0:
        issues.append("Kratkoročne obaveze su negativne.")
        severity = "high"

    if not is_missing(sales_revenue) and sales_revenue < 0:
        issues.append("Prihodi od prodaje su negativni.")
        severity = "medium" if severity != "high" else severity

    if not is_missing(data_quality_score) and data_quality_score < 70:
        issues.append("Kvalitet ekstrakcije je ispod 70.")
        severity = "medium" if severity != "high" else severity

    if altman_zone in ["Unknown", "", None]:
        issues.append("Altman Z' zona nije dostupna.")
        severity = "high"

    needs_review = severity in ["medium", "high"] or len(issues) > 0

    return {
        "file_name": row.get("file_name"),
        "company_name": row.get("company_name"),
        "registration_number": row.get("registration_number"),
        "report_year": row.get("report_year"),
        "altman_z_prime_score": row.get("altman_z_prime_score"),
        "altman_z_prime_zone": row.get("altman_z_prime_zone"),
        "data_quality_score": row.get("data_quality_score"),
        "severity": severity,
        "needs_review": needs_review,
        "issues": " | ".join(issues),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Nije pronađen fajl: {INPUT_PATH}. "
            "Prvo pokreni: python -m backend.build_dataset_from_pdfs"
        )

    df = pd.read_csv(INPUT_PATH)

    audit_rows = [audit_row(row) for _, row in df.iterrows()]
    audit_df = pd.DataFrame(audit_rows)

    AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_df.to_csv(AUDIT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    clean_df = df[audit_df["needs_review"] == False].copy()
    clean_df.to_csv(CLEAN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    total_rows = len(df)
    review_count = int(audit_df["needs_review"].sum())
    clean_count = total_rows - review_count

    print("Audit dataseta je završen.")
    print(f"Ukupno redova: {total_rows}")
    print(f"Kandidati za čisti dataset: {clean_count}")
    print(f"Za ručnu provjeru: {review_count}")
    print()
    print(f"Audit izvještaj: {AUDIT_OUTPUT_PATH}")
    print(f"Čisti kandidati: {CLEAN_OUTPUT_PATH}")
    print()

    if review_count > 0:
        print("Redovi za provjeru:")
        review_columns = [
            "file_name",
            "altman_z_prime_score",
            "altman_z_prime_zone",
            "data_quality_score",
            "issues",
        ]

        print(
            audit_df[audit_df["needs_review"] == True][review_columns]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()