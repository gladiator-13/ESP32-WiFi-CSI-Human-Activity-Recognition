
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "handcrafted_features"
    / "lolo"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "random_forest"
    / "lolo"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "random_forest"
    / "lolo"
)

RANDOM_STATE = 42
EXPECTED_FEATURES = 468
EXPECTED_CLASSES = np.arange(1, 21)

PARAMETER_GRID = {
    "max_depth": [None, 30],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5],
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_split(fold, split):

    folder = FEATURE_DIR / fold / split

    X = np.load(folder / "X_features.npy")
    y = np.load(folder / "y.npy")

    metadata = pd.read_csv(folder / "metadata.csv")

    assert X.ndim == 2
    assert X.shape[1] == EXPECTED_FEATURES
    assert len(X) == len(y) == len(metadata)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == set(EXPECTED_CLASSES)

    print(
        f"{fold.upper()} | {split.upper()}: "
        f"X={X.shape}, y={y.shape}"
    )

    return X, y, metadata


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
    }


# --------------------------------------------------
# TRAIN AND EVALUATE ONE FOLD
# --------------------------------------------------

def run_fold(fold):

    print("\n" + "=" * 70)
    print(f"RANDOM FOREST - {fold.upper()}")
    print("=" * 70)

    fold_model_dir = MODEL_DIR / fold
    fold_report_dir = REPORT_DIR / fold

    fold_model_dir.mkdir(parents=True, exist_ok=True)
    fold_report_dir.mkdir(parents=True, exist_ok=True)

    # Load all splits.
    X_train, y_train, _ = load_split(fold, "train")
    X_val, y_val, _ = load_split(fold, "validation")
    X_test, y_test, test_metadata = load_split(fold, "test")

    # --------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------

    candidates = list(ParameterGrid(PARAMETER_GRID))

    print(f"\nConfigurations: {len(candidates)}")

    results = []
    best_score = -1
    best_parameters = None

    for index, parameters in enumerate(candidates, start=1):

        print("\n" + "-" * 65)
        print(f"Candidate {index}/{len(candidates)}")
        print(f"Parameters: {parameters}")

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,
            **parameters
        )

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        predictions = model.predict(X_val)
        metrics = calculate_metrics(y_val, predictions)

        results.append({
            "candidate": index,
            "n_estimators": 300,
            **parameters,
            **metrics,
            "training_time_seconds": training_time,
        })

        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_parameters = parameters

    # Save validation search results.
    results_df = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False
    )

    results_df.to_csv(
        fold_report_dir / "validation_results.csv",
        index=False
    )

    print("\nBest parameters:")
    print(best_parameters)

    print(f"Best validation macro-F1: {best_score:.4f}")

    # --------------------------------------------------
    # REFIT ON TRAIN + VALIDATION
    # --------------------------------------------------

    print("\nRefitting selected model on train + validation...")

    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
        **best_parameters
    )

    start_time = time.time()
    final_model.fit(X_train_val, y_train_val)
    refit_time = time.time() - start_time

    # --------------------------------------------------
    # HELD-OUT TEST EVALUATION
    # --------------------------------------------------

    print("\nEvaluating on held-out location...")

    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\n" + "=" * 65)
    print(f"{fold.upper()} TEST RESULTS")
    print("=" * 65)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = fold_model_dir / "best_model.joblib"
    joblib.dump(final_model, model_path)

    # --------------------------------------------------
    # SAVE PARAMETERS AND METRICS
    # --------------------------------------------------

    with open(
        fold_report_dir / "best_parameters.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump({
            "model": "RandomForestClassifier",
            "fold": fold,
            "n_estimators": 300,
            "random_state": RANDOM_STATE,
            "class_weight": None,
            "best_parameters": best_parameters,
            "best_validation_macro_f1": float(best_score),
            "training_samples_after_refit": len(y_train_val),
            "number_of_features": EXPECTED_FEATURES,
            "number_of_classes": len(EXPECTED_CLASSES),
            "refit_time_seconds": refit_time,
        }, file, indent=4)

    with open(
        fold_report_dir / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {k: float(v) for k, v in test_metrics.items()},
            file,
            indent=4
        )

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y_test,
        test_predictions,
        labels=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        fold_report_dir / "test_classification_report.csv"
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=EXPECTED_CLASSES
    )

    pd.DataFrame(
        cm,
        index=[f"Actual_{i}" for i in EXPECTED_CLASSES],
        columns=[f"Predicted_{i}" for i in EXPECTED_CLASSES]
    ).to_csv(
        fold_report_dir / "test_confusion_matrix.csv"
    )

    # --------------------------------------------------
    # PREDICTIONS WITH METADATA
    # --------------------------------------------------

    predictions_df = test_metadata.copy()

    predictions_df["actual_label"] = y_test
    predictions_df["predicted_label"] = test_predictions
    predictions_df["correct"] = (
        y_test == test_predictions
    )

    predictions_df.to_csv(
        fold_report_dir / "test_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print(f"Model: {model_path}")
    print(f"Reports: {fold_report_dir}")

    return {
        "fold": fold,
        **test_metrics,
        "best_validation_macro_f1": best_score,
        "held_out_location": (
            test_metadata["location"].iloc[0]
            if "location" in test_metadata.columns
            else "unknown"
        ),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 70)
    print("RANDOM FOREST - LEAVE-ONE-LOCATION-OUT EXPERIMENT")
    print("=" * 70)

    all_results = []

    for fold in ["fold_1", "fold_2", "fold_3"]:

        result = run_fold(fold)
        all_results.append(result)

    # --------------------------------------------------
    # SAVE OVERALL RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "lolo_summary.csv",
        index=False
    )

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    summary = results_df[metric_columns].agg(
        ["mean", "std"]
    )

    summary.to_csv(
        REPORT_DIR / "lolo_mean_std.csv"
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST - LOLO SUMMARY")
    print("=" * 70)

    print(results_df.to_string(index=False))
    print("\nMean and standard deviation:")
    print(summary)

    print("\nRANDOM FOREST LOLO EXPERIMENT COMPLETED!")


if __name__ == "__main__":
    main()
