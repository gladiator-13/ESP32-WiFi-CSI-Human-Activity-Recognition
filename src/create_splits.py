
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_SEED = 42

ACTIVITIES = list(range(1, 21))

SPLITS = ["train", "validation", "test"]


# --------------------------------------------------
# 2. LOAD AND VALIDATE METADATA
# --------------------------------------------------

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {
        "recording_id",
        "participant",
        "activity",
        "recording_type",
        "location"
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    # Each participant must belong to exactly one location.
    location_counts = metadata.groupby(
        "participant"
    )["location"].nunique()

    if (location_counts != 1).any():
        raise ValueError(
            "Some participants are assigned to multiple locations."
        )

    return metadata


# --------------------------------------------------
# 3. CREATE PARTICIPANT-WISE SPLITS
# --------------------------------------------------

def create_splits(metadata):

    participants = (
        metadata[["participant", "location"]]
        .drop_duplicates()
        .sort_values(["location", "participant"])
    )

    rng = np.random.default_rng(RANDOM_SEED)

    assignments = []

    for location, group in participants.groupby("location"):

        people = group["participant"].to_numpy().copy()

        # Shuffle participants reproducibly.
        rng.shuffle(people)

        total = len(people)

        # Approximately 20% validation and 20% test.
        n_test = round(total * 0.20)
        n_validation = round(total * 0.20)

        # Keep at least one participant in each split.
        n_test = max(1, n_test)
        n_validation = max(1, n_validation)

        n_train = total - n_test - n_validation

        if n_train < 1:
            raise ValueError(
                f"Not enough participants in {location}"
            )

        test_people = people[:n_test]

        validation_people = people[
            n_test:n_test + n_validation
        ]

        train_people = people[
            n_test + n_validation:
        ]

        for person in train_people:
            assignments.append({
                "participant": person,
                "location": location,
                "split": "train"
            })

        for person in validation_people:
            assignments.append({
                "participant": person,
                "location": location,
                "split": "validation"
            })

        for person in test_people:
            assignments.append({
                "participant": person,
                "location": location,
                "split": "test"
            })

    return pd.DataFrame(assignments)


# --------------------------------------------------
# 4. VALIDATE PARTICIPANT ASSIGNMENTS
# --------------------------------------------------

def validate_participant_splits(split_df, metadata):

    expected_people = set(metadata["participant"].unique())
    assigned_people = set(split_df["participant"])

    # Every participant must be assigned exactly once.
    assert split_df["participant"].is_unique

    assert assigned_people == expected_people

    # Verify no participant leakage.
    train = set(
        split_df.loc[
            split_df["split"] == "train",
            "participant"
        ]
    )

    validation = set(
        split_df.loc[
            split_df["split"] == "validation",
            "participant"
        ]
    )

    test = set(
        split_df.loc[
            split_df["split"] == "test",
            "participant"
        ]
    )

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    print("Participant separation: PASSED")

    print("Total participants:", len(assigned_people))


# --------------------------------------------------
# 5. VALIDATE ACTIVITY CLASS COVERAGE
# --------------------------------------------------

def validate_activity_coverage(split_df, metadata):

    activity_metadata = metadata[
        metadata["recording_type"] == "activity"
    ].copy()

    recording_splits = activity_metadata.merge(
        split_df[["participant", "split"]],
        on="participant",
        how="left",
        validate="many_to_one"
    )

    if recording_splits["split"].isna().any():
        raise ValueError(
            "Some recordings have no split assignment."
        )

    class_coverage = pd.crosstab(
        recording_splits["split"],
        recording_splits["activity"]
    ).reindex(
        index=SPLITS,
        columns=ACTIVITIES,
        fill_value=0
    )

    print("\nActivity class coverage:")
    print(class_coverage)

    if (class_coverage == 0).any().any():
        raise ValueError(
            "At least one split is missing an activity class."
        )

    print("\nActivity coverage: PASSED")

    return class_coverage


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("Loading metadata...")

    metadata = load_metadata()

    print("Metadata recordings:", len(metadata))
    print(
        "Participants:",
        metadata["participant"].nunique()
    )

    print("\nCreating participant-wise splits...")

    split_df = create_splits(metadata)

    validate_participant_splits(split_df, metadata)

    class_coverage = validate_activity_coverage(
        split_df,
        metadata
    )

    # Add split assignment to every recording,
    # including baseline recordings.
    recording_splits = metadata.merge(
        split_df[["participant", "split"]],
        on="participant",
        how="left",
        validate="many_to_one"
    )

    if recording_splits["split"].isna().any():
        raise ValueError(
            "Some recordings were not assigned to a split."
        )

    # Create output directory.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save participant-level assignments.
    split_df.to_csv(
        OUTPUT_DIR / "within_location_split.csv",
        index=False
    )

    # Save recording-level assignments.
    recording_splits.to_csv(
        OUTPUT_DIR / "recording_splits.csv",
        index=False
    )

    # Save activity coverage.
    class_coverage.to_csv(
        OUTPUT_DIR / "activity_class_coverage.csv"
    )

    # Print participant distribution.
    print("\nParticipant distribution by location:")

    participant_summary = pd.crosstab(
        split_df["location"],
        split_df["split"]
    ).reindex(columns=SPLITS, fill_value=0)

    print(participant_summary)

    print("\nOverall participant distribution:")

    print(split_df["split"].value_counts())

    print("\nRecording distribution:")

    print(
        recording_splits.groupby(
            ["split", "recording_type"]
        ).size().unstack(fill_value=0)
    )

    print("\nAll split validations passed!")

    print("\nFiles saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
