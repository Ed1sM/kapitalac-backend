from pathlib import Path

import pandas as pd

from backend.ml_model import ML_FEATURE_NAMES


CLEAN_CANDIDATES_PATH = Path("data/processed/company_features_clean_candidates.csv")
OUTPUT_PATH = Path("data/processed/training_dataset.csv")
REVIEW_OUTPUT_PATH = Path("data/processed/training_dataset_review.csv")


def auto_label_from_altman(zone: str):
    """
    Automatsko označavanje na osnovu Altman Z' zone.

    0 = stabilna kompanija
    1 = rizična kompanija

    Safe zone se tretira kao stabilna.
    Distress zone se tretira kao rizična.
    Grey zone i Unknown se ne koriste za treniranje.
    """
    if not isinstance(zone, str):
        return ""

    zone = zone.strip().lower()

    if zone == "safe zone":
        return 0

    if zone == "distress zone":
        return 1

    return ""


def get_label_source(zone: str) -> str:
    """
    Objašnjava odakle je target vrijednost došla.
    """
    if not isinstance(zone, str):
        return "not_labeled"

    zone = zone.strip().lower()

    if zone == "safe zone":
        return "auto_altman_safe_zone"

    if zone == "distress zone":
        return "auto_altman_distress_zone"

    if zone == "grey zone":
        return "excluded_grey_zone"

    return "excluded_unknown"


def main():
    if not CLEAN_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Nije pronađen fajl: {CLEAN_CANDIDATES_PATH}. "
            "Prvo pokreni: python -m backend.audit_dataset"
        )

    df = pd.read_csv(CLEAN_CANDIDATES_PATH)

    result = pd.DataFrame()

    result["company_name"] = df.get("company_name")
    result["registration_number"] = df.get("registration_number")
    result["activity_code"] = df.get("activity_code")
    result["report_year"] = df.get("report_year")
    result["file_name"] = df.get("file_name")

    result["altman_z_prime_score"] = df.get("altman_z_prime_score")
    result["altman_z_prime_zone"] = df.get("altman_z_prime_zone")
    result["data_quality_score"] = df.get("data_quality_score")

    result["target_bankrupt"] = result["altman_z_prime_zone"].apply(
        auto_label_from_altman
    )

    result["label_source"] = result["altman_z_prime_zone"].apply(
        get_label_source
    )

    for feature_name in ML_FEATURE_NAMES:
        result[feature_name] = df.get(feature_name)

    output_columns = [
        "company_name",
        "registration_number",
        "activity_code",
        "report_year",
        "file_name",
        "target_bankrupt",
        "label_source",
        "altman_z_prime_score",
        "altman_z_prime_zone",
        "data_quality_score",
        *ML_FEATURE_NAMES,
    ]

    result = result[output_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    review_df = result[result["target_bankrupt"] == ""].copy()
    review_df.to_csv(REVIEW_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    usable_df = result[result["target_bankrupt"] != ""].copy()

    class_counts = usable_df["target_bankrupt"].value_counts().to_dict()

    print("Training dataset je automatski pripremljen.")
    print(f"Glavni fajl: {OUTPUT_PATH}")
    print(f"Kompanije koje nijesu korišćene za treniranje: {REVIEW_OUTPUT_PATH}")
    print()
    print("Raspodjela automatskih labela:")
    print(f"- 0 stabilna kompanija: {class_counts.get(0, 0)}")
    print(f"- 1 rizična kompanija: {class_counts.get(1, 0)}")
    print(f"- isključeno iz treninga: {len(review_df)}")
    print()
    print("Napomena:")
    print("Ovo su weak labels izvedene iz Altman Z' zone.")
    print("Safe zone = 0, Distress zone = 1, Grey zone se ne koristi za trening.")
    print()
    print("Pregled trening redova:")

    preview_columns = [
        "file_name",
        "company_name",
        "target_bankrupt",
        "label_source",
        "altman_z_prime_score",
        "altman_z_prime_zone",
        "data_quality_score",
    ]

    print(usable_df[preview_columns].to_string(index=False))


if __name__ == "__main__":
    main()