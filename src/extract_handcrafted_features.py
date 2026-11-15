
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "normalized"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "handcrafted_features"

SPLITS = ["train", "validation", "test"]

FEATURES = 52
BATCH_SIZE = 32

STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "rms",
    "mean_abs_diff",
]


def extract_features(batch):
    """
    Input shape: (batch_size, 200, 52)
    Output shape: (batch_size, 468)
    """

    mean = np.mean(batch, axis=1)
    std = np.std(batch, axis=1)
    minimum = np.min(batch, axis=1)
    maximum = np.max(batch, axis=1)
    median = np.median(batch, axis=1)

    q25 = np.percentile(batch, 25, axis=1)
    q75 = np.percentile(batch, 75, axis=1)

    rms = np.sqrt(np.mean(np.square(batch), axis=1))

    mean_abs_diff = np.mean(
        np.abs(np.diff(batch, axis=1)),
        axis=1
    )

    result = np.stack(
        [
            mean,
            std,
            minimum,
            maximum,
            median,
            q25,
            q75,
            rms,
            mean_abs_diff,
        ],
        axis=2
    )

    # Arrange as:
    # [subcarrier_0_mean, subcarrier_0_std, ...]
    # [subcarrier_1_mean, subcarrier_1_std, ...]

    return result.reshape(len(batch), -1).astype(np.float32)


def create_feature_names():

    names = []

    for subcarrier in range(FEATURES):

        for statistic in STATISTICS:

            names.append(
                f"subcarrier_{subcarrier}_{statistic}"
            )

    return names


def process_split(input_folder, output_folder):

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    X = np.load(
        input_folder / "X.npy",
        mmap_mode="r"
    )

    y = np.load(
        input_folder / "y.npy"
    )

    metadata = pd.read_csv(
        input_folder / "metadata.csv"
    )

    assert X.ndim == 3
    assert X.shape[1:] == (200, FEATURES)
    assert len(X) == len(y) == len(metadata)

    feature_count = FEATURES * len(STATISTICS)

    X_features = np.lib.format.open_memmap(
        output_folder / "X_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(X), feature_count)
    )

    print(
        f"Extracting features from {len(X)} windows..."
    )

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        batch = np.asarray(X[start:end])

        X_features[start:end] = extract_features(batch)

    X_features.flush()

    np.save(
        output_folder / "y.npy",
        y
    )

    metadata.to_csv(
        output_folder / "metadata.csv",
        index=False
    )

    pd.DataFrame(
        {"feature_name": create_feature_names()}
    ).to_csv(
        output_folder / "feature_names.csv",
        index=False
    )

    # Basic validation
    assert X_features.shape == (
        len(X),
        feature_count
    )

    assert np.isfinite(X_features).all()

    del X_features

    print(
        f"PASSED | Shape: ({len(X)}, {feature_count})"
    )


def process_dataset(input_folder, output_folder):

    print("\n" + "=" * 65)
    print(f"PROCESSING: {input_folder.name}")
    print("=" * 65)

    for split in SPLITS:

        print(f"\nSplit: {split}")

        process_split(
            input_folder / split,
            output_folder / split
        )


def main():

    print("=" * 65)
    print("HANDCRAFTED CSI FEATURE EXTRACTION")
    print("=" * 65)

    # Within-location
    process_dataset(
        INPUT_DIR / "within_location",
        OUTPUT_DIR / "within_location"
    )

    # Three independent LOLO folds
    for fold in [1, 2, 3]:

        process_dataset(
            INPUT_DIR / "lolo" / f"fold_{fold}",
            OUTPUT_DIR / "lolo" / f"fold_{fold}"
        )

    print("\n" + "=" * 65)
    print("ALL FEATURE EXTRACTION TASKS COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
