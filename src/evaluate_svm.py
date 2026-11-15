
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


ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "handcrafted_features" / "within_location"

MODEL_PATH = ROOT / "models" / "svm" / "within_location" / "best_model.joblib"

REPORT_DIR = ROOT / "reports" / "svm" / "within_location"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 65)
    print("SVM TEST EVALUATION")
    print("=" * 65)

    # Load untouched test data
    X_test = np.load(FEATURE_DIR / "test" / "X_features.npy")
    y_test = np.load(FEATURE_DIR / "test" / "y.npy")
    metadata = pd.read_csv(FEATURE_DIR / "test" / "metadata.csv")

    print(f"Test features: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    # Load trained SVM pipeline
    model = joblib.load(MODEL_PATH)

    # Predict
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\nTEST RESULTS")
    print("-" * 45)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")

    # Class labels
    labels = np.arange(1, 21)
    target_names = [f"Activity_{i:02d}" for i in labels]

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        REPORT_DIR / "test_classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names,
    )

    cm_df.to_csv(REPORT_DIR / "test_confusion_matrix.csv")

    # Overall metrics
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
    }])

    metrics_df.to_csv(
        REPORT_DIR / "test_metrics.csv",
        index=False,
    )

    # Save predictions with metadata
    predictions_df = metadata.copy()
    predictions_df["true_label"] = y_test
    predictions_df["predicted_label"] = y_pred
    predictions_df["correct"] = y_test == y_pred

    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nPer-class classification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))

    print("\nSaved evaluation reports to:")
    print(REPORT_DIR)

    print("\nSVM test evaluation completed successfully!")


if __name__ == "__main__":
    main()
