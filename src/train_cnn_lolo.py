
"""
Train and evaluate 1D-CNN models using Leave-One-Location-Out (LOLO)
cross-validation for ESP32 Wi-Fi CSI Human Activity Recognition.

Three independent models are trained:
    Fold 1: Train on Location 2 + Location 3; test on Location 1
    Fold 2: Train on Location 1 + Location 3; test on Location 2
    Fold 3: Train on Location 1 + Location 2; test on Location 3

Model selection is based on validation macro-F1.
Each held-out test set is loaded only after model selection.
"""

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "normalized"
    / "lolo"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "cnn"
    / "lolo"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "cnn"
    / "lolo"
)

NUM_CLASSES = 20
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001

CLASS_NAMES = [
    f"Activity_{i:02d}"
    for i in range(1, NUM_CLASSES + 1)
]

FOLDS = [1, 2, 3]


# ============================================================
# REPRODUCIBILITY AND CPU CONFIGURATION
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    print("TensorFlow threading already initialized.")


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 70)
print("1D-CNN HUMAN ACTIVITY RECOGNITION - LOLO")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(fold_dir, split_name):

    split_dir = fold_dir / split_name

    X = np.load(split_dir / "X.npy")
    y = np.load(split_dir / "y.npy")
    metadata = pd.read_csv(split_dir / "metadata.csv")

    print(f"\n{split_name.upper()} DATA")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Metadata shape:", metadata.shape)

    if X.ndim != 3:
        raise ValueError(
            f"{split_name}: X must be 3-dimensional."
        )

    if X.shape[1:] != (200, 52):
        raise ValueError(
            f"{split_name}: Unexpected input shape {X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: X and y lengths differ."
        )

    if len(X) != len(metadata):
        raise ValueError(
            f"{split_name}: X and metadata lengths differ."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: Non-finite values found."
        )

    if not np.isin(y, np.arange(1, 21)).all():
        raise ValueError(
            f"{split_name}: Invalid activity labels."
        )

    # TensorFlow sparse categorical crossentropy
    # expects labels in the range 0-19.
    y = y.astype(np.int32) - 1

    X = X.astype(np.float32, copy=False)

    print("Validation: PASSED")

    return X, y, metadata


# ============================================================
# BUILD 1D-CNN
# ============================================================

def build_model():

    inputs = keras.Input(
        shape=(200, 52),
        name="csi_input"
    )

    # Convolution block 1
    x = layers.Conv1D(
        filters=64,
        kernel_size=7,
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Convolution block 2
    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Convolution block 3
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Feature aggregation
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected classifier
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="activity_output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CSI_1D_CNN"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# VALIDATION MACRO-F1 CALLBACK
# ============================================================

class ValidationMacroF1(keras.callbacks.Callback):

    def __init__(self, X_val, y_val):

        super().__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.best_macro_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):

        logs = logs if logs is not None else {}

        probabilities = self.model.predict(
            self.X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        macro_f1 = f1_score(
            self.y_val,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0
        )

        logs["val_macro_f1"] = macro_f1

        self.best_macro_f1 = max(
            self.best_macro_f1,
            macro_f1
        )

        print(
            f" - val_macro_f1: {macro_f1:.4f}"
        )


# ============================================================
# EVALUATION HELPER
# ============================================================

def calculate_metrics(y_true, y_pred, prefix):

    return {
        f"{prefix}_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        f"{prefix}_macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        f"{prefix}_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        f"{prefix}_weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )
    }


# ============================================================
# SAVE CLASSIFICATION REPORT AND CONFUSION MATRIX
# ============================================================

def save_evaluation_reports(
    y_true,
    y_pred,
    report_dir,
    prefix
):

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        report_dir / f"{prefix}_classification_report.csv"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        report_dir / f"{prefix}_confusion_matrix.csv"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_true,
    y_pred,
    metadata,
    output_path
):

    prediction_df = metadata.reset_index(drop=True).copy()

    prediction_df["actual_label"] = y_true + 1
    prediction_df["predicted_label"] = y_pred + 1
    prediction_df["correct"] = y_true == y_pred

    prediction_df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# RUN ONE LOLO FOLD
# ============================================================

def run_fold(fold_number):

    print("\n\n" + "=" * 70)
    print(f"STARTING LOLO FOLD {fold_number}")
    print("=" * 70)

    # Reset random seeds for reproducibility.
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Release resources from the previous fold.
    keras.backend.clear_session()
    gc.collect()

    fold_dir = DATA_DIR / f"fold_{fold_number}"

    fold_model_dir = (
        MODEL_DIR / f"fold_{fold_number}"
    )

    fold_report_dir = (
        REPORT_DIR / f"fold_{fold_number}"
    )

    fold_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD TRAINING AND VALIDATION DATA
    # --------------------------------------------------------

    X_train, y_train, train_metadata = load_split(
        fold_dir,
        "train"
    )

    X_val, y_val, val_metadata = load_split(
        fold_dir,
        "validation"
    )

    print("\nHeld-out test data will remain untouched")
    print("until model selection is completed.")

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()

    print("\nMODEL SUMMARY")

    model.summary()

    with open(
        fold_report_dir / "model_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        model.summary(
            print_fn=lambda line: f.write(line + "\n")
        )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    best_model_path = (
        fold_model_dir / "best_model.keras"
    )

    macro_f1_callback = ValidationMacroF1(
        X_val,
        y_val
    )

    callbacks = [

        # Must come before callbacks monitoring val_macro_f1.
        macro_f1_callback,

        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),

        keras.callbacks.CSVLogger(
            str(
                fold_report_dir / "training_history.csv"
            )
        )
    ]

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"TRAINING FOLD {fold_number}")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nFold {fold_number} training completed.")

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print("\nLoading best validation macro-F1 model...")

    best_model = keras.models.load_model(
        best_model_path
    )

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    val_probabilities = best_model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    val_predictions = np.argmax(
        val_probabilities,
        axis=1
    )

    val_metrics = calculate_metrics(
        y_val,
        val_predictions,
        "validation"
    )

    best_epoch = int(
        np.argmax(
            history.history["val_macro_f1"]
        ) + 1
    )

    val_metrics.update({
        "model": "1D-CNN",
        "split": "lolo",
        "fold": fold_number,
        "best_epoch": best_epoch,
        "total_epochs": len(history.history["loss"])
    })

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number} VALIDATION RESULTS")
    print("=" * 70)

    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    with open(
        fold_report_dir / "validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            val_metrics,
            f,
            indent=4
        )

    save_evaluation_reports(
        y_val,
        val_predictions,
        fold_report_dir,
        "validation"
    )

    save_predictions(
        y_val,
        val_predictions,
        val_metadata,
        fold_report_dir / "validation_predictions.csv"
    )

    # --------------------------------------------------------
    # LOAD HELD-OUT TEST DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"LOADING HELD-OUT TEST DATA - FOLD {fold_number}")
    print("=" * 70)

    X_test, y_test, test_metadata = load_split(
        fold_dir,
        "test"
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_probabilities = best_model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        "test"
    )

    # Identify the held-out location from test metadata.
    if "location" in test_metadata.columns:

        held_out_locations = (
            test_metadata["location"].dropna().unique()
        )

        if len(held_out_locations) == 1:
            held_out_location = str(
                held_out_locations[0]
            )
        else:
            held_out_location = "Multiple/Unknown"

    else:
        held_out_location = "Unknown"

    test_metrics.update({
        "model": "1D-CNN",
        "split": "lolo",
        "fold": fold_number,
        "held_out_location": held_out_location,
        "best_epoch": best_epoch,
        "total_epochs": len(history.history["loss"])
    })

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number} TEST RESULTS")
    print("=" * 70)

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    with open(
        fold_report_dir / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=4
        )

    save_evaluation_reports(
        y_test,
        test_predictions,
        fold_report_dir,
        "test"
    )

    save_predictions(
        y_test,
        test_predictions,
        test_metadata,
        fold_report_dir / "test_predictions.csv"
    )

    # --------------------------------------------------------
    # SAVE CONFIGURATION
    # --------------------------------------------------------

    config = {

        "model": "1D-CNN",
        "split": "lolo",
        "fold": fold_number,
        "held_out_location": held_out_location,

        "input_shape": [200, 52],
        "num_classes": NUM_CLASSES,

        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,

        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,

        "random_seed": SEED,
        "class_weights": None,

        "model_selection_metric": "validation_macro_f1",

        "test_evaluated": True

    }

    with open(
        fold_report_dir / "configuration.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print("\nFold outputs saved successfully.")

    # Return only the information needed for the summary.
    return test_metrics


# ============================================================
# RUN ALL THREE FOLDS
# ============================================================

print("\nStarting three-fold LOLO CNN evaluation...")

fold_results = []

for fold_number in FOLDS:

    result = run_fold(fold_number)

    fold_results.append(result)


# ============================================================
# LOLO SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL LOLO CNN SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame(fold_results)

summary_df.to_csv(
    REPORT_DIR / "lolo_summary.csv",
    index=False
)

metric_columns = [
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1"
]

mean_values = summary_df[metric_columns].mean()
std_values = summary_df[metric_columns].std()

mean_std_df = pd.DataFrame({
    "metric": metric_columns,
    "mean": mean_values.values,
    "std": std_values.values
})

mean_std_df.to_csv(
    REPORT_DIR / "lolo_mean_std.csv",
    index=False
)

print("\nFOLD-WISE RESULTS")

print(
    summary_df[
        [
            "fold",
            "held_out_location",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1"
        ]
    ].to_string(index=False)
)

print("\nMEAN AND STANDARD DEVIATION")

print(mean_std_df.to_string(index=False))


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("1D-CNN LOLO EVALUATION COMPLETED")
print("=" * 70)

print("Models:", MODEL_DIR)
print("Reports:", REPORT_DIR)

print("\nAll three LOLO folds completed successfully.")
