
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "handcrafted_features"
)

SPLITS = ["train", "validation", "test"]

EXPECTED_FEATURES = 468
EXPECTED_CLASSES = set(range(1, 21))


def validate_split(folder, split):

    split_folder = folder / split

    X = np.load(
        split_folder / "X_features.npy",
        mmap_mode="r"
    )

    y = np.load(split_folder / "y.npy")

    metadata = pd.read_csv(
        split_folder / "metadata.csv"
    )

    feature_names = pd.read_csv(
        split_folder / "feature_names.csv"
    )

    # Shape checks
    assert X.ndim == 2
    assert X.shape[1] == EXPECTED_FEATURES

    # Length checks
    assert len(X) == len(y) == len(metadata)

    # Numerical checks
    assert np.isfinite(X).all()

    # Label checks
    assert np.array_equal(
        y,
        metadata["activity"].to_numpy(dtype=np.int64)
    )

    assert set(np.unique(y)) == EXPECTED_CLASSES

    # Feature-name checks
    assert len(feature_names) == EXPECTED_FEATURES
    assert feature_names["feature_name"].is_unique

    # Metadata checks
    assert metadata["recording_id"].notna().all()

    print(
        f"{split.upper()}: PASSED | "
        f"Shape: {X.shape} | "
        f"Classes: {len(np.unique(y))}"
    )

    del X


def validate_dataset(name, folder):

    print("\n" + "=" * 60)
    print(f"VALIDATING: {name}")
    print("=" * 60)

    for split in SPLITS:
        validate_split(folder, split)

    print(f"\n{name}: ALL CHECKS PASSED")


def main():

    print("=" * 60)
    print("HANDCRAFTED FEATURE QUALITY ASSURANCE")
    print("=" * 60)

    validate_dataset(
        "WITHIN-LOCATION",
        FEATURE_DIR / "within_location"
    )

    lolo_folds = sorted(
        [
            path.name
            for path in (FEATURE_DIR / "lolo").glob("fold_*")
            if path.is_dir()
        ]
    )

    for fold_name in lolo_folds:

        fold_number = fold_name.split("_")[-1]

        validate_dataset(
            f"LOLO FOLD {fold_number}",
            FEATURE_DIR / "lolo" / fold_name
        )


    print("\n" + "=" * 60)
    print("ALL FEATURE VALIDATIONS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
