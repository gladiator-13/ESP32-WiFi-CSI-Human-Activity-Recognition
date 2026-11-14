from pathlib import Path
import re
import pandas as pd

# Project root / data / data / P00 ... P27
DATASET_DIR = Path("data/data")
OUTPUT_DIR = Path("data/processed")


def get_location(participant):
    participant_number = int(participant[1:])

    if 0 <= participant_number <= 9:
        return "Location_1"
    elif 10 <= participant_number <= 18:
        return "Location_2"
    elif 19 <= participant_number <= 27:
        return "Location_3"

    return "Unknown"


def extract_metadata(file_path):
    filename = file_path.stem

    # Check standard activity format: P02_A01_R1
    activity_match = re.search(r"(P\d+)_A(\d+)_R(\d+)", filename)
    if activity_match:
        participant = activity_match.group(1)
        return {
            "participant": participant,
            "recording_type": "activity",
            "activity": int(activity_match.group(2)),
            "repetition": int(activity_match.group(3)),
            "location": get_location(participant)
        }

    # Check baseline format: P02_BASE_R1_1790676738
    base_match = re.search(r"(P\d+)_BASE_R(\d+)", filename)
    if base_match:
        participant = base_match.group(1)
        return {
            "participant": participant,
            "recording_type": "baseline",
            "activity": 0,
            "repetition": int(base_match.group(2)),
            "location": get_location(participant)
        }

    return None


def build_dataset_index():
    if not DATASET_DIR.exists():
        print(f"Error: Folder not found at {DATASET_DIR}")
        return pd.DataFrame()

    records = []
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))

    for file_path in csv_files:
        metadata = extract_metadata(file_path)

        if metadata is None:
            print("Skipped:", file_path.name)
            continue

        records.append({
            "filename": file_path.name,
            "filepath": str(file_path.resolve()),
            **metadata
        })

    index = pd.DataFrame(records)

    if index.empty:
        print("No valid CSV files found!")
        return index

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "dataset_index.csv"
    index.to_csv(output_file, index=False)

    print("\n========== DATASET SUMMARY ==========")
    print("Total recordings:", len(index))

    print("\nRecording type distribution:")
    print(index["recording_type"].value_counts())

    print("\nLocation distribution:")
    print(index["location"].value_counts())

    print("\nParticipant distribution:")
    print(index["participant"].value_counts().sort_index())

    print("\nActivity distribution (0 = Baseline):")
    print(index["activity"].value_counts().sort_index())

    print("\nParticipants missing baseline files (expected 2):")
    for participant in sorted(index["participant"].unique()):
        base_count = len(
            index.loc[
                (index["participant"] == participant) & (index["recording_type"] == "baseline")
            ]
        )
        if base_count < 2:
            print(f"{participant} has {base_count} baseline file(s)")

    print(f"\nSaved: {output_file}")

    # 1. Recordings by participant and recording type
    print("\nRECORDING TYPE BY PARTICIPANT")
    print(
        index.pivot_table(
            index="participant",
            columns="recording_type",
            values="filename",
            aggfunc="count",
            fill_value=0
        )
    )

    # 2. Activity counts by location
    print("\nACTIVITY COUNTS BY LOCATION")
    activity_location = pd.crosstab(
        index["activity"],
        index["location"]
    )
    print(activity_location)

    # 3. Number of recordings for each participant/activity
    print("\nPARTICIPANT-ACTIVITY DISTRIBUTION")
    participant_activity = pd.crosstab(
        index["participant"],
        index["activity"]
    )
    print(participant_activity)

    # 4. Check duplicate repetitions
    duplicate_check = index[
        index.duplicated(
            subset=["participant", "activity", "repetition"],
            keep=False
        )
    ].sort_values(
        ["participant", "activity", "repetition"]
    )
    print("\nDUPLICATE PARTICIPANT-ACTIVITY-REPETITION COMBINATIONS")
    print(duplicate_check[
        ["participant", "activity", "repetition", "filename"]
    ])

    # 5. Check activity coverage (20 activities: 1-20)
    expected_activities = set(range(1, 21))
    print("\nMISSING ACTIVITY CHECK")
    for participant in sorted(index["participant"].unique()):
        found = set(
            index.loc[
                (index["participant"] == participant) &
                (index["recording_type"] == "activity"),
                "activity"
            ]
        )
        missing = sorted(expected_activities - found)
        if missing:
            print(participant, "missing:", missing)
        else:
            print(participant, "complete")

    return index


if __name__ == "__main__":
    index = build_dataset_index()
    if not index.empty:
        print("\nFirst 10 records:")
        print(index.head(10).to_string(index=False))
