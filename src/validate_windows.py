
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "windows"

WINDOW_SIZE = 200
FEATURES = 52
EXPECTED_WINDOWS_PER_RECORDING = 19

ACTIVITY_CLASSES = set(range(1, 21))


def validate_one_split(metadata, split_csv, split_name, expected_fold=None):

    folder = WINDOWS_DIR / split_csv / split_name

    X = np.load(folder / "X.npy", mmap_mode="r")
    y = np.load(folder / "y.npy", mmap_mode="r")
    window_meta = pd.read_csv(folder / "metadata.csv")

    # Check array dimensions
    assert X.ndim == 3
    assert X.shape[1:] == (WINDOW_SIZE, FEATURES)
    assert y.ndim == 1

    # Check matching lengths
    assert len(X) == len(y) == len(window_meta)

    # Check numerical values
    assert np.isfinite(X).all()
    assert set(np.unique(y)).issubset(ACTIVITY_CLASSES)

    # Check labels against metadata
    assert np.array_equal(
        y,
        window_meta["activity"].to_numpy(dtype=np.int64)
    )

    # Check window count for every recording
    counts = window_meta.groupby("recording_id").size()

    assert (counts == EXPECTED_WINDOWS_PER_RECORDING).all()

    # Check expected recordings
    expected_recordings = set(
        metadata.loc[
            metadata["split"] == split_name,
            "recording_id"
        ].astype(str)
    )

    actual_recordings = set(
        window_meta["recording_id"].astype(str)
    )

    assert expected_recordings == actual_recordings

    # Check window numbering and sample positions
    for recording_id, group in window_meta.groupby("recording_id"):

        group = group.sort_values("window_number")

        assert group["window_number"].tolist() == list(range(1, 20))

        expected_starts = list(range(0, 1900, 100))
        assert group["start_sample"].tolist() == expected_starts

        assert (
            group["end_sample"] - group["start_sample"] == 200
        ).all()

    if expected_fold is not None:
        assert (window_meta["fold"] == expected_fold).all()

    print(
        f"{split_name}: PASSED "
        f"| {len(X)} windows "
        f"| {len(counts)} recordings "
        f"| Shape: {X.shape}"
    )

    return window_meta


def validate_dataset(split_file, dataset_folder, fold=None):

    split_path = SPLITS_DIR / split_file
    source = pd.read_csv(split_path)

    # Keep only the current LOLO fold
    if fold is not None:
        source = source[source["fold"] == fold].copy()

    # Exclude baseline recordings (activity 0)
    source = source[
        source["activity"].isin(ACTIVITY_CLASSES)
    ].copy()

    all_metadata = []

    for split_name in ["train", "validation", "test"]:

        result = validate_one_split(
            source,
            dataset_folder,
            split_name,
            expected_fold=fold,
        )

        all_metadata.append(result)

    combined = pd.concat(all_metadata, ignore_index=True)

    # Participant separation
    participant_counts = (
        combined.groupby("participant")["split"].nunique()
    )

    assert (participant_counts == 1).all()

    # Every activity should be present in every split
    for split_name in ["train", "validation", "test"]:

        labels = set(
            combined.loc[
                combined["split"] == split_name,
                "activity"
            ].unique()
        )

        assert labels == ACTIVITY_CLASSES

    # Check source and generated recording counts
    assert len(combined) == len(source) * EXPECTED_WINDOWS_PER_RECORDING

    if fold is not None:

        test_location = source["test_location"].iloc[0]

        test_locations = set(
            combined.loc[
                combined["split"] == "test",
                "location"
            ].unique()
        )

        assert test_locations == {test_location}

        non_test_locations = set(
            combined.loc[
                combined["split"].isin(["train", "validation"]),
                "location"
            ].unique()
        )

        assert test_location not in non_test_locations

    print("\nDataset validation: PASSED")
    print(f"Total windows: {len(combined)}")
    print(f"Unique recordings: {combined['recording_id'].nunique()}")
    print(f"Unique participants: {combined['participant'].nunique()}")


def main():

    print("=" * 55)
    print("WINDOW DATASET QUALITY ASSURANCE")
    print("=" * 55)

    print("\nWITHIN-LOCATION")

    validate_dataset(
        "recording_splits.csv",
        "within_location",
    )

    print("\nLOLO")

    for fold in sorted(
        pd.read_csv(SPLITS_DIR / "lolo_recording_splits.csv")["fold"].unique()
    ):

        print(f"\nFOLD {fold}")

        validate_dataset(
            "lolo_recording_splits.csv",
            f"lolo/fold_{fold}",
            fold=fold,
        )

    print("\n" + "=" * 55)
    print("ALL WINDOW VALIDATIONS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
