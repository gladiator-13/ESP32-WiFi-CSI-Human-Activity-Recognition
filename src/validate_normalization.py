
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "normalized"
)

SPLITS = ["train", "validation", "test"]

WINDOW_SIZE = 200
FEATURES = 52
BATCH_SIZE = 32


def calculate_statistics(X):

    total = 0
    feature_sum = np.zeros(FEATURES, dtype=np.float64)
    feature_sum_sq = np.zeros(FEATURES, dtype=np.float64)

    for start in range(0, len(X), BATCH_SIZE):

        batch = np.asarray(
            X[start:start + BATCH_SIZE],
            dtype=np.float64
        )

        assert np.isfinite(batch).all()

        batch = batch.reshape(-1, FEATURES)

        feature_sum += batch.sum(axis=0)
        feature_sum_sq += np.square(batch).sum(axis=0)

        total += len(batch)

    mean = feature_sum / total

    variance = feature_sum_sq / total - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0))

    return mean, std


def validate_dataset(dataset_name, dataset_folder):

    print("\n" + "=" * 65)
    print(f"VALIDATING: {dataset_name}")
    print("=" * 65)

    scaler_path = dataset_folder / "scaler.joblib"

    assert scaler_path.exists(), "Scaler file missing!"

    scaler = joblib.load(scaler_path)

    assert len(scaler.mean_) == FEATURES
    assert len(scaler.scale_) == FEATURES

    print("Scaler dimensions: PASSED")

    for split in SPLITS:

        folder = dataset_folder / split

        X = np.load(
            folder / "X.npy",
            mmap_mode="r"
        )

        y = np.load(folder / "y.npy")

        metadata = pd.read_csv(
            folder / "metadata.csv"
        )

        assert X.shape[1:] == (WINDOW_SIZE, FEATURES)
        assert len(X) == len(y) == len(metadata)

        assert np.array_equal(
            y,
            metadata["activity"].to_numpy(dtype=np.int64)
        )

        mean, std = calculate_statistics(X)

        assert np.isfinite(mean).all()
        assert np.isfinite(std).all()

        if split == "train":

            max_mean_error = np.max(np.abs(mean))
            max_std_error = np.max(np.abs(std - 1))

            assert max_mean_error < 1e-4
            assert max_std_error < 1e-3

            print(
                f"\n{split.upper()}: PASSED "
                f"| Shape: {X.shape}"
            )

            print(
                f"Maximum absolute mean: "
                f"{max_mean_error:.8f}"
            )

            print(
                f"Maximum standard deviation error: "
                f"{max_std_error:.8f}"
            )

        else:

            print(
                f"{split.upper()}: PASSED "
                f"| Shape: {X.shape}"
            )

        del X

    print(f"\n{dataset_name}: ALL CHECKS PASSED")


def main():

    print("=" * 65)
    print("NORMALIZATION QUALITY ASSURANCE")
    print("=" * 65)

    validate_dataset(
        "WITHIN-LOCATION",
        NORMALIZED_DIR / "within_location"
    )

    lolo_folds = sorted(
        [
            path.name
            for path in (NORMALIZED_DIR / "lolo").glob("fold_*")
            if path.is_dir()
        ]
    )

    for fold_name in lolo_folds:

        fold_number = fold_name.split("_")[-1]

        validate_dataset(
            f"LOLO FOLD {fold_number}",
            NORMALIZED_DIR / "lolo" / fold_name
        )

    print("\n" + "=" * 65)
    print("ALL NORMALIZATION VALIDATIONS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
