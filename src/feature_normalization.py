
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "windows"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "normalized"

SPLITS = ["train", "validation", "test"]

FEATURES = 52
BATCH_SIZE = 32


def fit_training_scaler(train_folder):
    """
    Fit StandardScaler using training data only.

    Each CSI feature is normalized independently
    across all training windows and time samples.
    """

    X_train = np.load(
        train_folder / "X.npy",
        mmap_mode="r"
    )

    assert X_train.ndim == 3
    assert X_train.shape[2] == FEATURES

    scaler = StandardScaler()

    print(f"Fitting scaler on {X_train.shape} training windows...")

    for start in range(0, len(X_train), BATCH_SIZE):

        batch = X_train[start:start + BATCH_SIZE]

        batch = batch.reshape(-1, FEATURES)

        scaler.partial_fit(batch)

    print("Scaler fitted successfully.")

    return scaler


def transform_split(input_folder, output_folder, scaler):

    output_folder.mkdir(parents=True, exist_ok=True)

    X = np.load(
        input_folder / "X.npy",
        mmap_mode="r"
    )

    y = np.load(
        input_folder / "y.npy",
        mmap_mode="r"
    )

    metadata = pd.read_csv(
        input_folder / "metadata.csv"
    )

    assert X.shape[2] == FEATURES
    assert len(X) == len(y) == len(metadata)

    output_path = output_folder / "X.npy"

    X_normalized = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=X.shape
    )

    print(f"Transforming {len(X)} windows...")

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        batch = X[start:end]

        original_shape = batch.shape

        batch = batch.reshape(-1, FEATURES)

        transformed = scaler.transform(batch)

        X_normalized[start:end] = transformed.reshape(
            original_shape
        ).astype(np.float32)

    X_normalized.flush()

    # Preserve labels and metadata
    np.save(output_folder / "y.npy", y)

    shutil.copy2(
        input_folder / "metadata.csv",
        output_folder / "metadata.csv"
    )

    # Validate saved output
    assert X_normalized.shape == X.shape
    assert np.isfinite(X_normalized).all()

    del X_normalized

    print(
        f"PASSED | {len(X)} windows | "
        f"Shape: {X.shape}"
    )


def normalize_dataset(input_folder, output_folder):

    print("\n" + "=" * 65)
    print(f"NORMALIZING: {input_folder.name}")
    print("=" * 65)

    # Fit exclusively on training data
    scaler = fit_training_scaler(
        input_folder / "train"
    )

    # Save the fitted scaler
    output_folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        output_folder / "scaler.joblib"
    )

    # Transform train, validation and test
    for split in SPLITS:

        print(f"\nProcessing {split}...")

        transform_split(
            input_folder / split,
            output_folder / split,
            scaler
        )

    print("\nNormalization completed successfully.")

    print("\nTraining feature means (first 5):")
    print(np.round(scaler.mean_[:5], 4))

    print("\nTraining feature standard deviations (first 5):")
    print(np.round(scaler.scale_[:5], 4))


def main():

    print("=" * 65)
    print("CSI FEATURE NORMALIZATION")
    print("=" * 65)

    # 1. Within-location
    normalize_dataset(
        WINDOWS_DIR / "within_location",
        OUTPUT_DIR / "within_location"
    )

    # 2. LOLO: independent scaler for every fold
    lolo_folds = sorted(
        [
            path.name
            for path in (WINDOWS_DIR / "lolo").glob("fold_*")
            if path.is_dir()
        ]
    )

    for fold_name in lolo_folds:

        normalize_dataset(
            WINDOWS_DIR / "lolo" / fold_name,
            OUTPUT_DIR / "lolo" / fold_name
        )


    print("\n" + "=" * 65)
    print("ALL NORMALIZATION TASKS COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
