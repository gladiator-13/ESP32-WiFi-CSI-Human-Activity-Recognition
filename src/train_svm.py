from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    / "within_location"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "svm"
    / "within_location"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "svm"
    / "within_location"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
EXPECTED_FEATURES = 468
EXPECTED_CLASSES = set(range(1, 21))


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_split(split):

    folder = FEATURE_DIR / split

    X = np.load(folder / "X_features.npy")
    y = np.load(folder / "y.npy")

    assert X.ndim == 2
    assert X.shape[1] == EXPECTED_FEATURES
    assert len(X) == len(y)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == EXPECTED_CLASSES

    print(
        f"{split.upper()} loaded: "
        f"X={X.shape}, y={y.shape}"
    )

    return X, y


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("SVM - WITHIN-LOCATION EXPERIMENT")
    print("=" * 65)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    # Candidate hyperparameters.
    parameter_grid = {
        "svm__C": [1, 10],
        "svm__gamma": ["scale", 0.001, 0.01],
    }

    candidates = list(ParameterGrid(parameter_grid))

    print(f"\nTotal configurations: {len(candidates)}")

    results = []

    best_score = -1
    best_parameters = None
    best_pipeline = None

    # --------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------

    for index, parameters in enumerate(candidates, start=1):

        print("\n" + "-" * 65)
        print(f"Training candidate {index}/{len(candidates)}")
        print(f"Parameters: {parameters}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    **{
                        key.replace("svm__", ""): value
                        for key, value in parameters.items()
                    }
                )
            )
        ])

        start_time = time.time()

        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time

        metrics = evaluate_model(
            pipeline,
            X_val,
            y_val
        )

        result = {
            "candidate": index,
            "kernel": "rbf",
            "C": parameters["svm__C"],
            "gamma": parameters["svm__gamma"],
            **metrics,
            "training_time_seconds": training_time,
        }

        results.append(result)

        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

        if metrics["macro_f1"] > best_score:

            best_score = metrics["macro_f1"]
            best_parameters = parameters
            best_pipeline = pipeline

    # --------------------------------------------------
    # SAVE VALIDATION RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False
    )

    results_path = REPORT_DIR / "validation_results.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("SVM HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 65)

    print("\nTop configurations:")

    print(
        results_df[
            [
                "candidate",
                "C",
                "gamma",
                "accuracy",
                "macro_f1",
            ]
        ].head()
    )

    print("\nBest parameters:")
    print(best_parameters)

    print(f"Best validation macro-F1: {best_score:.4f}")

    # --------------------------------------------------
    # REFIT ON TRAIN + VALIDATION
    # --------------------------------------------------

    print("\nRefitting the selected SVM on train + validation...")

    X_train_val = np.concatenate(
        [X_train, X_val],
        axis=0
    )

    y_train_val = np.concatenate(
        [y_train, y_val],
        axis=0
    )

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=best_parameters["svm__C"],
                gamma=best_parameters["svm__gamma"],
                class_weight=None,
                random_state=RANDOM_STATE,
            )
        )
    ])

    final_pipeline.fit(
        X_train_val,
        y_train_val
    )

    # --------------------------------------------------
    # SAVE MODEL AND PARAMETERS
    # --------------------------------------------------

    model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(
        final_pipeline,
        model_path
    )

    parameters_path = REPORT_DIR / "best_parameters.json"

    with open(parameters_path, "w") as file:

        json.dump(
            {
                "model": "SVC",
                "kernel": "rbf",
                "random_state": RANDOM_STATE,
                "class_weight": None,
                "best_parameters": {
                    "C": best_parameters["svm__C"],
                    "gamma": best_parameters["svm__gamma"],
                },
                "best_validation_macro_f1": float(best_score),
                "training_samples_after_refit": len(y_train_val),
                "number_of_features": EXPECTED_FEATURES,
                "number_of_classes": len(EXPECTED_CLASSES),
                "scaler": "StandardScaler fitted within Pipeline",
            },
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("FINAL SVM MODEL SAVED")
    print("=" * 65)

    print(f"Model: {model_path}")
    print(f"Parameters: {parameters_path}")
    print(f"Validation results: {results_path}")

    print(f"\nTraining samples used for final refit: {len(y_train_val)}")
    print(f"Number of features: {EXPECTED_FEATURES}")
    print(f"Number of classes: {len(EXPECTED_CLASSES)}")

    print("\nSVM TRAINING COMPLETED!")


if __name__ == "__main__":
    main()