
"""
Evaluate the trained CNN-BiLSTM-Attention model on the
within-location test set.

This script does NOT train or modify the model.
It loads the best validation-selected model and evaluates
it once on the untouched test set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model_layers import AdditiveTemporalAttention


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "normalized"
    / "within_location"
    / "test"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "cnn_bilstm_attention"
    / "within_location"
    / "best_model.keras"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "cnn_bilstm_attention"
    / "within_location"
)

NUM_CLASSES = 20
BATCH_SIZE = 128

CLASS_IDS = np.arange(NUM_CLASSES)

CLASS_NAMES = [
    f"Activity_{i:02d}"
    for i in range(1, NUM_CLASSES + 1)
]


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 70)
print("CNN-BiLSTM-ATTENTION WITHIN-LOCATION TEST EVALUATION")
print("=" * 70)

print("TensorFlow:", tf.__version__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(TEST_DIR / "X.npy")
y_test = np.load(TEST_DIR / "y.npy")

metadata = pd.read_csv(
    TEST_DIR / "metadata.csv"
)

print("X shape:", X_test.shape)
print("y shape:", y_test.shape)
print("Metadata shape:", metadata.shape)

if X_test.ndim != 3 or X_test.shape[1:] != (200, 52):
    raise ValueError(
        f"Unexpected test input shape: {X_test.shape}"
    )

if len(X_test) != len(y_test):
    raise ValueError("X_test and y_test lengths differ.")

if len(X_test) != len(metadata):
    raise ValueError(
        "Test data and metadata lengths differ."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Non-finite values found in test data."
    )

if not np.isin(y_test, np.arange(1, 21)).all():
    raise ValueError(
        "Invalid activity labels in test data."
    )

required_metadata = [
    "recording_id",
    "participant",
    "location",
    "activity",
    "window_number"
]

missing_columns = [
    column
    for column in required_metadata
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Missing metadata columns: {missing_columns}"
    )

# Labels are stored as 1-20 in the dataset.
# The model uses zero-based labels 0-19.
y_test = y_test.astype(np.int32) - 1

X_test = X_test.astype(
    np.float32,
    copy=False
)

print("Test data validation: PASSED")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AdditiveTemporalAttention": AdditiveTemporalAttention
    },
    compile=False
)

print("Model loaded successfully.")
print("Model:", model.name)


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)

confidence = np.max(
    probabilities,
    axis=1
)

print("Predictions generated:", len(predictions))


# ============================================================
# CALCULATE TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    predictions
)

test_macro_precision = precision_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="weighted",
    zero_division=0
)

metrics = {
    "model": "CNN-BiLSTM-Attention",
    "split": "within_location_test",
    "accuracy": float(test_accuracy),
    "macro_precision": float(test_macro_precision),
    "macro_recall": float(test_macro_recall),
    "macro_f1": float(test_macro_f1),
    "weighted_f1": float(test_weighted_f1),
    "test_samples": int(len(y_test)),
    "test_recordings": int(metadata["recording_id"].nunique()),
    "number_of_classes": NUM_CLASSES
}


# ============================================================
# PRINT AND SAVE TEST METRICS
# ============================================================

print("\n" + "=" * 70)
print("CNN-BiLSTM-ATTENTION TEST RESULTS")
print("=" * 70)

for key, value in metrics.items():
    print(f"{key}: {value}")

metrics_path = REPORT_DIR / "test_metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    predictions,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_path = (
    REPORT_DIR / "test_classification_report.csv"
)

pd.DataFrame(report).transpose().to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=CLASS_IDS
)

cm_path = (
    REPORT_DIR / "test_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(cm_path)


# ============================================================
# SAVE TEST PREDICTIONS WITH METADATA
# ============================================================

prediction_df = metadata.copy()

prediction_df["actual_label"] = y_test + 1
prediction_df["predicted_label"] = predictions + 1

prediction_df["actual_activity"] = [
    CLASS_NAMES[label]
    for label in y_test
]

prediction_df["predicted_activity"] = [
    CLASS_NAMES[label]
    for label in predictions
]

prediction_df["confidence"] = confidence

prediction_df["correct"] = (
    y_test == predictions
)

predictions_path = (
    REPORT_DIR / "test_predictions.csv"
)

prediction_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(metrics_path)
print(report_path)
print(cm_path)
print(predictions_path)

print("\nTest samples:", len(y_test))

print(
    "Unique recordings:",
    metadata["recording_id"].nunique()
)

print(
    "Correct predictions:",
    int(np.sum(y_test == predictions))
)

print(
    "Incorrect predictions:",
    int(np.sum(y_test != predictions))
)

print("\nModel training was NOT performed.")
print("The trained model was NOT modified.")
print("Test evaluation completed successfully.")
