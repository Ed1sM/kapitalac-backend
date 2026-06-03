from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/company_features_from_pdfs.csv")
AUDIT_OUTPUT_PATH = Path("data/processed/dataset_audit_report.csv")
CLEAN_OUTPUT_PATH = Path("data/processed/company_features_clean_candidates.csv")


def is_missing(value) -> bool:
    if pd.isna(value):
        return True

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


def add_issue(issues: list[str], issue: str):
    issues.append(issue)


def audit_row(row: pd.Series) -> dict:
    issues = []

    file_name = row.get("file_name")
    company_name = row.get("company_name")

    total_assets = row.get("total_assets")
    equity = row.get("equity")
    total_liabilities = row.get("total_liabilities")
    current_assets = row.get("current_assets")
    short_term_liabilities = row.get("short_term_liabilities")
    sales_revenue = row.get("sales_revenue")
    altman_z_prime_score = row.get("altman_z_prime_score")
    altman_z_prime_zone = row.get("altman_z_prime_zone")
    data_quality_score = row.get("data_quality_score")
    debt_to_assets = row.get("debt_to_assets")
    roa = row.get("roa")
    asset_turnover = row.get("asset_turnover")
    working_capital_to_assets = row.get("working_capital_to_assets")

    required_fields = [
        "total_assets",
        "equity",
        "current_assets",
        "short_term_liabilities",
        "sales_revenue",
        "operating_profit",
        "retained_earnings_to_assets",
    ]

    for field in required_fields:
        if is_missing(row.get(field)):
            add_issue(issues, f"Nedostaje obavezno polje: {field}")

    if not is_missing(total_assets) and total_assets <= 0:
        add_issue(issues, "Ukupna aktiva je manja ili jednaka nuli")

    if not is_missing(total_assets) and not is_missing(equity):
        if equity > total_assets * 1.5:
            add_issue(
                issues,
                "Kapital je neuobičajeno veći od ukupne aktive; moguće pogrešno očitan total_assets ili equity",
            )

    if not is_missing(total_liabilities) and total_liabilities < 0:
        add_issue(
            issues,
            "Ukupne obaveze su negativne; moguće pogrešno očitan total_assets ili equity",
        )

    if not is_missing(current_assets) and not is_missing(total_assets):
        if current_assets > total_assets * 1.2:
            add_issue(
                issues,
                "Obrtna sredstva su neuobičajeno veća od ukupne aktive",
            )

    if not is_missing(short_term_liabilities) and short_term_liabilities < 0:
        add_issue(issues, "Kratkoročne obaveze su negativne")

    if not is_missing(sales_revenue) and sales_revenue < 0:
        add_issue(issues, "Prihodi od prodaje su negativni")

    if not is_missing(asset_turnover) and asset_turnover > 20:
        add_issue(
            issues,
            "Obrt aktive je ekstremno visok; moguće pogrešno očitana ukupna aktiva ili prihod",
        )

    if not is_missing(roa) and abs(roa) > 1:
        add_issue(
            issues,
            "ROA je ekstreman; moguće pogrešno očitana neto dobit ili ukupna aktiva",
        )

    if not is_missing(debt_to_assets) and (debt_to_assets < 0 or debt_to_assets > 3):
        add_issue(
            issues,
            "Debt-to-assets je van realnog raspona; provjeriti aktivu, kapital i obaveze",
        )

    if not is_missing(working_capital_to_assets) and abs(working_capital_to_assets) > 5:
        add_issue(
            issues,
            "Working capital / assets je ekstreman; provjeriti obrtna sredstva i ukupnu aktivu",
        )

    if is_missing(altman_z_prime_score) or altman_z_prime_zone == "Unknown":
        add_issue(issues, "Altman Z' nije izračunat")

    if not is_missing(data_quality_score) and data_quality_score < 70:
        add_issue(issues, "Nizak ili srednji kvalitet ekstrakcije")

    needs_review = len(issues) > 0

    return {
        "file_name": file_name,
        "company_name": company_name,
        "altman_z_prime_score": altman_z_prime_score,
        "altman_z_prime_zone": altman_z_prime_zone,
        "data_quality_score": data_quality_score,
        "needs_review": needs_review,
        "issues_count": len(issues),
        "issues": " | ".join(issues),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Nije pronađen fajl: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    audit_rows = []

    for _, row in df.iterrows():
        audit_rows.append(audit_row(row))

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    clean_candidates = audit_df[audit_df["needs_review"] == False]
    clean_file_names = clean_candidates["file_name"].tolist()

    clean_df = df[df["file_name"].isin(clean_file_names)].copy()
    clean_df.to_csv(CLEAN_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    total_rows = len(df)
    review_count = int(audit_df["needs_review"].sum())
    clean_count = total_rows - review_count

    print("Audit završen.")
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