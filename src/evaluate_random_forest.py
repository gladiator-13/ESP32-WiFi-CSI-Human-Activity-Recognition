
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "handcrafted_features"
    / "within_location"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest"
    / "within_location"
    / "best_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "random_forest"
    / "within_location"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = np.arange(1, 21)


def main():

    print("=" * 65)
    print("RANDOM FOREST - FINAL TEST EVALUATION")
    print("=" * 65)

    # Load the saved model
    print("\nLoading trained model...")
    model = joblib.load(MODEL_PATH)

    # Load untouched test data
    print("Loading test data...")

    test_folder = FEATURE_DIR / "test"

    X_test = np.load(test_folder / "X_features.npy")
    y_test = np.load(test_folder / "y.npy")

    metadata = pd.read_csv(
        test_folder / "metadata.csv"
    )

    # Basic checks
    assert X_test.ndim == 2
    assert X_test.shape[1] == 468
    assert len(X_test) == len(y_test) == len(metadata)
    assert np.isfinite(X_test).all()
    assert np.array_equal(
        y_test,
        metadata["activity"].to_numpy(dtype=np.int64)
    )

    print(f"Test samples: {len(y_test)}")
    print(f"Test features: {X_test.shape[1]}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 65)
    print("OVERALL TEST RESULTS")
    print("=" * 65)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")

    # Per-class classification report
    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = REPORT_DIR / "test_classification_report.csv"
    report_df.to_csv(report_path)

    print("\n" + "=" * 65)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 65)

    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Predicted_{c}" for c in CLASSES]
    )

    cm_path = REPORT_DIR / "test_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # Save overall metrics
    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "test_samples": len(y_test),
        "number_of_features": X_test.shape[1],
        "number_of_classes": len(CLASSES),
    }

    metrics_df = pd.DataFrame([metrics])

    metrics_path = REPORT_DIR / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save predictions alongside window metadata
    prediction_df = metadata.copy()
    prediction_df["actual_activity"] = y_test
    prediction_df["predicted_activity"] = predictions

    prediction_path = REPORT_DIR / "test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    print("\n" + "=" * 65)
    print("TEST EVALUATION FILES SAVED")
    print("=" * 65)

    print(f"Metrics: {metrics_path}")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"Predictions: {prediction_path}")

    print("\nFINAL TEST EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
