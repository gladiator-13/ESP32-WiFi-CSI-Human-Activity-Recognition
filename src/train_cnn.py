
"""
Train a 1D-CNN for ESP32 Wi-Fi CSI Human Activity Recognition.

Dataset:
    data/processed/normalized/within_location/

Input:
    X: (N, 200, 52)
    y: (N,) with labels 1-20

Model selection:
    Validation macro-F1

The test set is intentionally not loaded in this script.
"""

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
    / "within_location"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "cnn"
    / "within_location"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "cnn"
    / "within_location"
)

NUM_CLASSES = 20
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001

CLASS_NAMES = [
    f"Activity_{i:02d}"
    for i in range(1, NUM_CLASSES + 1)
]


# ============================================================
# REPRODUCIBILITY AND CPU CONFIGURATION
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# Limit CPU thread usage to avoid excessive contention.
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    print("TensorFlow threading already initialized.")

print("=" * 65)
print("1D-CNN HUMAN ACTIVITY RECOGNITION")
print("=" * 65)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split_name):

    split_dir = DATA_DIR / split_name

    X = np.load(split_dir / "X.npy")
    y = np.load(split_dir / "y.npy")

    print(f"\n{split_name.upper()} DATA")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim != 3:
        raise ValueError(f"{split_name}: X must be 3D.")

    if X.shape[1:] != (200, 52):
        raise ValueError(
            f"{split_name}: Unexpected input shape {X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: X and y lengths differ."
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

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("\nTest data will remain untouched.")


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


model = build_model()

print("\nMODEL SUMMARY")
model.summary()

with open(
    REPORT_DIR / "model_summary.txt",
    "w",
    encoding="utf-8"
    ) as f:
        model.summary(
            print_fn=lambda line: f.write(line + "\n")
        )


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
            f" — val_macro_f1: {macro_f1:.4f}"
        )


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "best_model.keras"

macro_f1_callback = ValidationMacroF1(
    X_val,
    y_val
)

callbacks = [

    # Must come before callbacks that monitor val_macro_f1.
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
        str(REPORT_DIR / "training_history.csv")
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 65)
print("STARTING TRAINING")
print("=" * 65)

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

print("\nTraining completed.")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best validation macro-F1 model...")

best_model = keras.models.load_model(
    best_model_path
)

val_probabilities = best_model.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0
)

val_predictions = np.argmax(
    val_probabilities,
    axis=1
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_macro_precision = precision_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_macro_recall = recall_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_weighted_f1 = f1_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

metrics = {
    "model": "1D-CNN",
    "split": "within_location",
    "validation_accuracy": float(val_accuracy),
    "validation_macro_precision": float(val_macro_precision),
    "validation_macro_recall": float(val_macro_recall),
    "validation_macro_f1": float(val_macro_f1),
    "validation_weighted_f1": float(val_weighted_f1),
    "best_epoch": int(
        np.argmax(history.history["val_macro_f1"]) + 1
    ),
    "total_epochs": len(history.history["loss"])
}

print("\n" + "=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

for key, value in metrics.items():
    print(f"{key}: {value}")

with open(REPORT_DIR / "validation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_val,
    val_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "validation_classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=np.arange(NUM_CLASSES)
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    REPORT_DIR / "validation_confusion_matrix.csv"
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "actual_label": y_val + 1,

    "predicted_label": val_predictions + 1,

    "correct": y_val == val_predictions

})

prediction_df.to_csv(
    REPORT_DIR / "validation_predictions.csv",
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "model": "1D-CNN",

    "input_shape": [200, 52],

    "num_classes": NUM_CLASSES,

    "optimizer": "Adam",

    "learning_rate": LEARNING_RATE,

    "batch_size": BATCH_SIZE,

    "maximum_epochs": EPOCHS,

    "early_stopping_patience": 15,

    "random_seed": SEED,

    "class_weights": None,

    "model_selection_metric": "validation_macro_f1",

    "test_evaluated": False

}

with open(REPORT_DIR / "configuration.json", "w") as f:
    json.dump(config, f, indent=4)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("1D-CNN TRAINING COMPLETED")
print("=" * 65)

print("Best model:", best_model_path)
print("Reports:", REPORT_DIR)

print("\nTest set was NOT used.")
print("All training outputs saved successfully.")
