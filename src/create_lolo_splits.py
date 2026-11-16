from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_SEED = 42
ACTIVITY_CLASSES = list(range(1, 21))

SPLIT_NAMES = ["train", "validation", "test"]


# --------------------------------------------------
# Load and validate metadata
# --------------------------------------------------

def load_metadata():
    print("Loading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {
        "recording_id",
        "participant",
        "activity",
        "recording_type",
        "location",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    if metadata["participant"].isna().any():
        raise ValueError("Participant column contains missing values.")

    if metadata["location"].isna().any():
        raise ValueError("Location column contains missing values.")

    participant_locations = (
        metadata.groupby("participant")["location"].nunique()
    )

    if (participant_locations != 1).any():
        raise ValueError(
            "A participant is associated with multiple locations."
        )

    print(f"Total recordings: {len(metadata)}")
    print(f"Total participants: {metadata['participant'].nunique()}")

    return metadata


# --------------------------------------------------
# Create one LOLO fold
# --------------------------------------------------

def create_one_fold(metadata, test_location, fold_number, rng):

    locations = sorted(metadata["location"].unique())

    if len(locations) < 2:
        raise ValueError(
            f"Expected at least 2 locations, found {len(locations)}."
        )

    train_locations = [
        location for location in locations
        if location != test_location
    ]

    participant_table = (
        metadata[["participant", "location"]]
        .drop_duplicates()
    )

    assignments = []

    # Held-out location: test only
    test_participants = participant_table[
        participant_table["location"] == test_location
    ]["participant"].tolist()

    for participant in test_participants:
        assignments.append({
            "participant": participant,
            "location": test_location,
            "split": "test",
            "fold": fold_number,
        })

    # Remaining two locations: train and validation
    for location in train_locations:

        participants = participant_table[
            participant_table["location"] == location
        ]["participant"].tolist()

        participants = sorted(participants)
        rng.shuffle(participants)

        # Reserve approximately 20% for validation
        n_validation = max(1, round(len(participants) * 0.20))

        validation_participants = participants[:n_validation]
        train_participants = participants[n_validation:]

        for participant in train_participants:
            assignments.append({
                "participant": participant,
                "location": location,
                "split": "train",
                "fold": fold_number,
            })

        for participant in validation_participants:
            assignments.append({
                "participant": participant,
                "location": location,
                "split": "validation",
                "fold": fold_number,
            })

    return pd.DataFrame(assignments)


# --------------------------------------------------
# Validate one fold
# --------------------------------------------------

def validate_fold(metadata, assignments, test_location, fold_number):

    print(f"\nValidating Fold {fold_number}...")

    # Every participant must occur exactly once
    if assignments["participant"].duplicated().any():
        raise ValueError(
            f"Fold {fold_number}: duplicate participant assignments."
        )

    # Every participant must be assigned
    expected = set(metadata["participant"].unique())
    actual = set(assignments["participant"].unique())

    if expected != actual:
        raise ValueError(
            f"Fold {fold_number}: participant assignment mismatch."
        )

    # Test participants must belong only to the held-out location
    test_rows = assignments[assignments["split"] == "test"]

    if set(test_rows["location"].unique()) != {test_location}:
        raise ValueError(
            f"Fold {fold_number}: incorrect test location."
        )

    # No held-out location in train or validation
    non_test_rows = assignments[
        assignments["split"].isin(["train", "validation"])
    ]

    if test_location in set(non_test_rows["location"]):
        raise ValueError(
            f"Fold {fold_number}: test location leaked into training."
        )

    # Every activity must be represented in each split
    activity_metadata = metadata[
        metadata["recording_type"] == "activity"
    ]

    merged = activity_metadata.merge(
        assignments[["participant", "split"]],
        on="participant",
        how="left",
        validate="many_to_one",
    )

    coverage = pd.crosstab(
        merged["split"],
        merged["activity"],
    )

    coverage = coverage.reindex(
        index=SPLIT_NAMES,
        columns=ACTIVITY_CLASSES,
        fill_value=0,
    )

    if (coverage == 0).any().any():
        raise ValueError(
            f"Fold {fold_number}: one or more activities are missing."
        )

    print("Participant separation: PASSED")
    print("Held-out location isolation: PASSED")
    print("Activity coverage: PASSED")

    return coverage


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    metadata = load_metadata()

    locations = sorted(metadata["location"].unique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_participant_splits = []
    all_recording_splits = []
    all_coverage = []

    rng = np.random.default_rng(RANDOM_SEED)

    for fold_number, test_location in enumerate(locations, start=1):

        print("\n" + "=" * 55)
        print(f"FOLD {fold_number}: TEST = {test_location}")
        print("=" * 55)

        assignments = create_one_fold(
            metadata,
            test_location,
            fold_number,
            rng,
        )

        coverage = validate_fold(
            metadata,
            assignments,
            test_location,
            fold_number,
        )

        assignments["test_location"] = test_location

        # Participant-level assignments
        all_participant_splits.append(assignments)

        # Recording-level assignments, including baseline
        recording_splits = metadata.merge(
            assignments,
            on=["participant", "location"],
            how="left",
            validate="many_to_one",
        )

        if recording_splits["split"].isna().any():
            raise ValueError(
                f"Fold {fold_number}: some recordings were not assigned."
            )

        all_recording_splits.append(recording_splits)

        # Save coverage with fold information
        coverage = coverage.reset_index()
        coverage.insert(0, "fold", fold_number)
        coverage.insert(1, "test_location", test_location)

        all_coverage.append(coverage)

        print("\nParticipant distribution:")
        print(
            assignments.groupby(
                ["location", "split"]
            ).size().unstack(fill_value=0)
        )

        print("\nOverall participant distribution:")
        print(assignments["split"].value_counts())

        print("\nRecording distribution:")
        print(
            recording_splits.groupby(
                ["split", "recording_type"]
            ).size().unstack(fill_value=0)
        )

    # Combine all folds
    participant_output = pd.concat(
        all_participant_splits,
        ignore_index=True,
    )

    recording_output = pd.concat(
        all_recording_splits,
        ignore_index=True,
    )

    coverage_output = pd.concat(
        all_coverage,
        ignore_index=True,
    )

    # Save outputs
    participant_output.to_csv(
        OUTPUT_DIR / "lolo_participant_splits.csv",
        index=False,
    )

    recording_output.to_csv(
        OUTPUT_DIR / "lolo_recording_splits.csv",
        index=False,
    )

    coverage_output.to_csv(
        OUTPUT_DIR / "lolo_activity_class_coverage.csv",
        index=False,
    )

    print("\n" + "=" * 55)
    print(f"ALL {len(locations)} LOLO FOLDS COMPLETED SUCCESSFULLY!")
    print("=" * 55)

    print("\nSaved files:")
    print("1. lolo_participant_splits.csv")
    print("2. lolo_recording_splits.csv")
    print("3. lolo_activity_class_coverage.csv")


if __name__ == "__main__":
    main()