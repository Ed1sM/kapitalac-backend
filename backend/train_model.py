from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml_model import ML_FEATURE_NAMES


DATASET_PATH = Path("data/processed/training_dataset.csv")
MODEL_OUTPUT_PATH = Path("models/kapitalac_ml_model.joblib")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Nije pronađen dataset: {DATASET_PATH}. "
            "Prvo pokreni: python -m backend.prepare_training_dataset"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["target_bankrupt", *ML_FEATURE_NAMES]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Nedostaju kolone u datasetu: {missing_columns}")

    df = df.copy()

    df["target_bankrupt"] = pd.to_numeric(
        df["target_bankrupt"],
        errors="coerce",
    )

    before_count = len(df)
    df = df.dropna(subset=["target_bankrupt"])
    after_count = len(df)

    dropped_count = before_count - after_count

    if dropped_count > 0:
        print(f"Izbačeno redova bez target_bankrupt vrijednosti: {dropped_count}")

    df["target_bankrupt"] = df["target_bankrupt"].astype(int)

    invalid_targets = sorted(
        set(df["target_bankrupt"].unique().tolist()) - {0, 1}
    )

    if invalid_targets:
        raise ValueError(
            "Kolona target_bankrupt smije sadržati samo 0 ili 1. "
            f"Neispravne vrijednosti: {invalid_targets}"
        )

    for feature_name in ML_FEATURE_NAMES:
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    if len(df) < 10:
        raise ValueError(
            "Premalo označenih redova za treniranje. "
            "Potrebno je bar 10 označenih firmi, a bolje 30+."
        )

    unique_targets = sorted(df["target_bankrupt"].dropna().unique().tolist())

    if unique_targets != [0, 1]:
        raise ValueError(
            "Dataset mora sadržati obje klase: 0 i 1. "
            f"Trenutno pronađene klase: {unique_targets}"
        )

    class_counts = df["target_bankrupt"].value_counts().to_dict()

    print("Raspodjela klasa:")
    print(f"- 0 stabilna firma: {class_counts.get(0, 0)}")
    print(f"- 1 rizična firma: {class_counts.get(1, 0)}")

    if class_counts.get(0, 0) < 5 or class_counts.get(1, 0) < 5:
        print()
        print("UPOZORENJE:")
        print("Jedna klasa ima manje od 5 primjera.")
        print("Model će raditi, ali rezultati neće biti dovoljno pouzdani.")
        print()


def calculate_metrics(y_true, y_pred, y_probability) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    try:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_probability), 4)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def train_candidate_models(X_train, X_test, y_train, y_test) -> list[dict]:
    models = []

    logistic_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    candidates = [
        ("Logistic Regression", logistic_model),
        ("Random Forest", random_forest_model),
    ]

    for model_name, model in candidates:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_probability = model.predict_proba(X_test)[:, 1]
        else:
            y_probability = y_pred

        metrics = calculate_metrics(y_test, y_pred, y_probability)

        models.append(
            {
                "name": model_name,
                "model": model,
                "metrics": metrics,
            }
        )

    return models


def choose_best_model(models: list[dict]) -> dict:
    def score_model(model_info: dict):
        metrics = model_info["metrics"]

        roc_auc = metrics.get("roc_auc")
        f1 = metrics.get("f1", 0)

        if roc_auc is not None:
            return roc_auc

        return f1

    return max(models, key=score_model)


def main():
    raw_df = load_dataset()
    df = prepare_dataset(raw_df)
    validate_dataset(df)

    X = df[ML_FEATURE_NAMES]
    y = df["target_bankrupt"].astype(int)

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    trained_models = train_candidate_models(X_train, X_test, y_train, y_test)
    best_model = choose_best_model(trained_models)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model_name": best_model["name"],
        "model_type": "machine_learning",
        "model": best_model["model"],
        "feature_names": ML_FEATURE_NAMES,
        "metrics": best_model["metrics"],
    }

    joblib.dump(model_bundle, MODEL_OUTPUT_PATH)

    print()
    print("Treniranje završeno.")
    print(f"Najbolji model: {best_model['name']}")
    print(f"Model sačuvan u: {MODEL_OUTPUT_PATH}")
    print()
    print("Metrike:")
    for metric_name, metric_value in best_model["metrics"].items():
        print(f"- {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()