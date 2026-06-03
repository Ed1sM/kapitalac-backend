from pathlib import Path

import pandas as pd

from backend.ml_model import ML_FEATURE_NAMES


def main():
    output_path = Path("data/processed/training_dataset_template.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "company_name",
        "registration_number",
        "report_year",
        "target_bankrupt",
        *ML_FEATURE_NAMES,
    ]

    example_rows = [
        {
            "company_name": "Primjer stabilne firme",
            "registration_number": "00000001",
            "report_year": 2025,
            "target_bankrupt": 0,
            "current_ratio": 2.5,
            "quick_ratio": 1.3,
            "cash_ratio": 0.4,
            "debt_to_assets": 0.25,
            "debt_to_equity": 0.33,
            "roa": 0.08,
            "operating_margin": 0.12,
            "net_profit_margin": 0.09,
            "asset_turnover": 1.1,
            "working_capital_to_assets": 0.35,
            "retained_earnings_to_assets": 0.45,
            "ebit_to_assets": 0.10,
            "equity_to_liabilities": 3.0,
            "sales_to_assets": 1.1,
        },
        {
            "company_name": "Primjer rizične firme",
            "registration_number": "00000002",
            "report_year": 2025,
            "target_bankrupt": 1,
            "current_ratio": 0.7,
            "quick_ratio": 0.3,
            "cash_ratio": 0.05,
            "debt_to_assets": 0.85,
            "debt_to_equity": 5.5,
            "roa": -0.06,
            "operating_margin": -0.04,
            "net_profit_margin": -0.08,
            "asset_turnover": 0.6,
            "working_capital_to_assets": -0.20,
            "retained_earnings_to_assets": -0.10,
            "ebit_to_assets": -0.04,
            "equity_to_liabilities": 0.18,
            "sales_to_assets": 0.6,
        },
    ]

    df = pd.DataFrame(example_rows, columns=columns)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Šablon dataseta je kreiran: {output_path}")
    print("Kopiraj ovaj fajl kao data/processed/training_dataset.csv i popuni ga stvarnim podacima.")


if __name__ == "__main__":
    main()