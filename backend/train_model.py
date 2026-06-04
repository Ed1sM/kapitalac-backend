import os
from pathlib import Path
from datetime import datetime

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

from backend.config import MLFLOW_ALLOW_FILE_STORE, MLFLOW_TRACKING_URI
from backend.ml_model import ML_FEATURE_NAMES


# MLflow file-store mora biti dozvoljen prije korišćenja MLflow-a.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = MLFLOW_ALLOW_FILE_STORE


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False


DATASET_PATH = Path("data/processed/training_dataset.csv")
MODEL_OUTPUT_PATH = Path("models/kapitalac_ml_model.joblib")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Nije pronađen dataset: {DATASET_PATH}. "
            "Prvo pokreni: python -m backend.prepare_training_dataset"
        )

    return pd.read_csv(DATASET_PATH)


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

    invalid_targets = sorted(set(df["target_bankrupt"].unique().tolist()) - {0, 1})

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


def calculate_reference_stats(df: pd.DataFrame) -> dict:
    stats = {}

    for feature_name in ML_FEATURE_NAMES:
        series = pd.to_numeric(df[feature_name], errors="coerce").dropna()

        if series.empty:
            continue

        stats[feature_name] = {
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std(ddof=0)), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "q05": round(float(series.quantile(0.05)), 6),
            "q95": round(float(series.quantile(0.95)), 6),
        }

    return stats


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

    if XGBOOST_AVAILABLE:
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())

        scale_pos_weight = 1

        if positive_count > 0:
            scale_pos_weight = max(1, negative_count / positive_count)

        xgboost_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        candidates.append(("XGBoost", xgboost_model))
    else:
        print("UPOZORENJE: xgboost nije instaliran, XGBoost model se preskače.")

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
    preference = {
        "XGBoost": 3,
        "Random Forest": 2,
        "Logistic Regression": 1,
    }

    def score_model(model_info: dict):
        metrics = model_info["metrics"]
        roc_auc = metrics.get("roc_auc")
        f1 = metrics.get("f1", 0)

        primary_score = roc_auc if roc_auc is not None else f1

        return (
            primary_score,
            f1,
            preference.get(model_info["name"], 0),
        )

    return max(models, key=score_model)


def log_to_mlflow(best_model: dict, all_models: list[dict], training_info: dict):
    """
    MLflow koristimo za MLOps tracking:
    - parametri
    - metrike
    - poređenje kandidata
    - informacije o treningu

    Model ne logujemo kroz MLflow registry.
    Model se čuva stabilno preko joblib-a.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow nije dostupan. Preskačem MLflow logovanje.")
        return None

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("kapitalac-bankruptcy-risk")

        with mlflow.start_run(run_name=f"kapitalac-{best_model['name']}") as run:
            mlflow.log_param("best_model", best_model["name"])
            mlflow.log_param("model_storage", str(MODEL_OUTPUT_PATH))
            mlflow.log_param("feature_count", len(ML_FEATURE_NAMES))
            mlflow.log_param("training_rows", training_info["training_rows"])
            mlflow.log_param("test_rows", training_info["test_rows"])
            mlflow.log_param("total_labeled_rows", training_info["total_labeled_rows"])
            mlflow.log_param("class_0_count", training_info["class_counts"].get(0, 0))
            mlflow.log_param("class_1_count", training_info["class_counts"].get(1, 0))
            mlflow.log_param("labeling_method", training_info["labeling_method"])
            mlflow.log_param("model_artifact_note", "Model is stored with joblib, not MLflow registry.")

            for metric_name, metric_value in best_model["metrics"].items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)

            for model_info in all_models:
                prefix = model_info["name"].lower().replace(" ", "_")

                for metric_name, metric_value in model_info["metrics"].items():
                    if metric_value is not None:
                        mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

            return run.info.run_id

    except Exception as error:
        print()
        print("UPOZORENJE: MLflow logovanje nije uspjelo.")
        print(f"Razlog: {error}")
        print("Treniranje se nastavlja i model će biti sačuvan normalno.")
        print()
        return None


def main():
    raw_df = load_dataset()
    df = prepare_dataset(raw_df)
    validate_dataset(df)

    X = df[ML_FEATURE_NAMES]
    y = df["target_bankrupt"].astype(int)

    class_counts = y.value_counts().to_dict()
    can_stratify = y.value_counts().min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    trained_models = train_candidate_models(X_train, X_test, y_train, y_test)
    best_model = choose_best_model(trained_models)

    reference_stats = calculate_reference_stats(df)

    training_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_labeled_rows": int(len(df)),
        "class_counts": {
            int(key): int(value)
            for key, value in class_counts.items()
        },
        "labeling_method": "weak_supervision_altman_z_prime",
        "note": (
            "Model je treniran na weak labelama izvedenim iz Altman Z' zona. "
            "Safe zone = 0, Distress zone = 1, Grey/Unknown se ne koriste za trening."
        ),
    }

    mlflow_run_id = log_to_mlflow(
        best_model=best_model,
        all_models=trained_models,
        training_info=training_info,
    )

    training_info["mlflow_run_id"] = mlflow_run_id
    training_info["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model_name": best_model["name"],
        "model_type": "machine_learning_hybrid",
        "model": best_model["model"],
        "feature_names": ML_FEATURE_NAMES,
        "metrics": best_model["metrics"],
        "all_model_metrics": {
            model_info["name"]: model_info["metrics"]
            for model_info in trained_models
        },
        "training_info": training_info,
        "reference_stats": reference_stats,
        "hybrid_note": (
            "Trenirani su Logistic Regression, Random Forest i XGBoost. "
            "Kao aktivni model automatski se bira kandidat sa najboljim metrikama. "
            "XGBoost ostaje dio hibridnog poređenja i evaluacije."
        ),
    }

    joblib.dump(model_bundle, MODEL_OUTPUT_PATH)

    print()
    print("Treniranje završeno.")
    print(f"Najbolji model: {best_model['name']}")
    print(f"Model sačuvan u: {MODEL_OUTPUT_PATH}")
    print()
    print("Metrike najboljeg modela:")
    for metric_name, metric_value in best_model["metrics"].items():
        print(f"- {metric_name}: {metric_value}")

    print()
    print("Svi kandidati:")
    for model_info in trained_models:
        print(f"- {model_info['name']}: {model_info['metrics']}")

    print()
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    if mlflow_run_id:
        print(f"MLflow run ID: {mlflow_run_id}")
    else:
        print("MLflow run ID: nije kreiran, ali model je uspješno sačuvan.")


if __name__ == "__main__":
    main()