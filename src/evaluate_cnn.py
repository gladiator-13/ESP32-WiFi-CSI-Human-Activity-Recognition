
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "normalized" / "within_location" / "test"
MODEL_PATH = ROOT / "models" / "cnn" / "within_location" / "best_model.keras"
OUTPUT_DIR = ROOT / "reports" / "cnn" / "within_location"

NUM_CLASSES = 20
CLASS_LABELS = [f"Activity_{i:02d}" for i in range(1, 21)]


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    print("\nLoading test data...")

    X_test = np.load(DATA_DIR / "X.npy")
    y_test = np.load(DATA_DIR / "y.npy")

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    assert X_test.ndim == 3
    assert X_test.shape[1:] == (200, 52)
    assert len(X_test) == len(y_test)
    assert np.isfinite(X_test).all()
    assert set(np.unique(y_test)) == set(range(1, 21))

    # Convert labels from 1–20 to 0–19 for TensorFlow.
    y_test_encoded = y_test.astype(np.int64) - 1

    print("Test data validation: PASSED")

    return X_test.astype(np.float32), y_test_encoded, y_test


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("\nLoading saved CNN model...")

    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, X_test, y_test_encoded, y_test_original):

    print("\nGenerating test predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=64,
        verbose=1
    )

    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred = y_pred_encoded + 1

    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    macro_precision = precision_score(
        y_test_encoded, y_pred_encoded,
        average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_test_encoded, y_pred_encoded,
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test_encoded, y_pred_encoded,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test_encoded, y_pred_encoded,
        average="weighted", zero_division=0
    )

    results = {
        "model": "1D-CNN",
        "split": "within_location",
        "evaluation": "test",
        "test_accuracy": float(accuracy),
        "test_macro_precision": float(macro_precision),
        "test_macro_recall": float(macro_recall),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_samples": int(len(y_test_encoded)),
    }

    print("\n" + "=" * 65)
    print("CNN TEST RESULTS")
    print("=" * 65)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # Per-class classification report
    report = classification_report(
        y_test_encoded,
        y_pred_encoded,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        OUTPUT_DIR / "test_classification_report.csv",
        index=True
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test_encoded,
        y_pred_encoded,
        labels=list(range(NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_LABELS,
        columns=CLASS_LABELS
    )

    cm_df.to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    # Save predictions with original labels
    predictions_df = pd.DataFrame({
        "actual_label": y_test_original,
        "actual_activity": [
            f"Activity_{label:02d}" for label in y_test_original
        ],
        "predicted_label": y_pred,
        "predicted_activity": [
            f"Activity_{label:02d}" for label in y_pred
        ],
        "confidence": np.max(probabilities, axis=1),
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # Save overall metrics
    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    print("\nPer-class classification report:")
    print(report_df.round(4).to_string())

    print("\nSaved evaluation outputs:")
    print(OUTPUT_DIR / "test_metrics.json")
    print(OUTPUT_DIR / "test_classification_report.csv")
    print(OUTPUT_DIR / "test_confusion_matrix.csv")
    print(OUTPUT_DIR / "test_predictions.csv")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test_encoded, y_test_original = load_test_data()
    model = load_model()

    assert model.output_shape[-1] == NUM_CLASSES

    evaluate(
        model,
        X_test,
        y_test_encoded,
        y_test_original
    )

    print("\nCNN test evaluation completed successfully.")


if __name__ == "__main__":
    main()
