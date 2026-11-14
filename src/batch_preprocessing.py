from pathlib import Path
import re
import numpy as np
import pandas as pd

from preprocessing import preprocess_recording


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROCESSED_DIR / "signals"

SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Participant-to-location mapping
# --------------------------------------------------

def get_location(participant):
    participant_number = int(participant[1:])

    if 0 <= participant_number <= 9:
        return "Location_1"
    elif 10 <= participant_number <= 18:
        return "Location_2"
    elif 19 <= participant_number <= 27:
        return "Location_3"
    else:
        return "Unknown"


# --------------------------------------------------
# 3. Extract metadata from filename
# Expected example:
# P00_A01_R1_1790671306.csv
# --------------------------------------------------

def extract_metadata(file_path):
    pattern = r"^(P\d+)_(A(\d+)|BASE)_R(\d+)_.*\.csv$"

    match = re.match(pattern, file_path.name)

    if not match:
        raise ValueError(
            f"Filename does not match expected format: {file_path.name}"
        )

    participant = match.group(1)
    activity_label = match.group(2)
    repetition = int(match.group(4))

    # BASE recordings represent activity 0.
    activity = (
        0 if activity_label == "BASE"
        else int(match.group(3))
    )

    recording_type = (
        "baseline" if activity == 0 else "activity"
    )

    return {
        "recording_id": file_path.stem,
        "participant": participant,
        "activity": activity,
        "repetition": repetition,
        "recording_type": recording_type,
        "location": get_location(participant),
        "raw_file": str(file_path.relative_to(PROJECT_ROOT)),
    }

# --------------------------------------------------
# 4. Process all recordings
# --------------------------------------------------

csv_files = sorted(RAW_DATA_DIR.glob("P*/*.csv"))

print(f"Total CSV files found: {len(csv_files)}")

successful_records = []
failed_records = []

for index, file_path in enumerate(csv_files, start=1):

    try:
        metadata = extract_metadata(file_path)

        # Run the preprocessing pipeline you already tested
        processed_signal = preprocess_recording(file_path)

        # Confirm expected dimensions
        if processed_signal.shape != (2000, 52):
            raise ValueError(
                f"Unexpected output shape: {processed_signal.shape}"
            )

        # Save processed signal
        output_path = SIGNALS_DIR / f"{file_path.stem}.npy"
        np.save(output_path, processed_signal)

        metadata["processed_file"] = str(
            output_path.relative_to(PROJECT_ROOT)
        )

        successful_records.append(metadata)

    except Exception as error:

        failed_records.append({
            "raw_file": str(file_path.relative_to(PROJECT_ROOT)),
            "error": str(error),
        })

    if index % 25 == 0 or index == len(csv_files):
        print(
            f"Progress: {index}/{len(csv_files)} | "
            f"Successful: {len(successful_records)} | "
            f"Failed: {len(failed_records)}"
        )


# --------------------------------------------------
# 5. Save reports
# --------------------------------------------------

metadata_df = pd.DataFrame(successful_records)
errors_df = pd.DataFrame(failed_records)

metadata_path = PROCESSED_DIR / "metadata.csv"
errors_path = PROCESSED_DIR / "preprocessing_errors.csv"

metadata_df.to_csv(metadata_path, index=False)
errors_df.to_csv(errors_path, index=False)


# --------------------------------------------------
# 6. Final summary
# --------------------------------------------------

print("\n========== BATCH PREPROCESSING SUMMARY ==========")

print(f"Total files found: {len(csv_files)}")
print(f"Successfully processed: {len(successful_records)}")
print(f"Failed: {len(failed_records)}")

if not metadata_df.empty:
    print("\nRecording type counts:")
    print(metadata_df["recording_type"].value_counts())

    print("\nLocation counts:")
    print(metadata_df["location"].value_counts())

print(f"\nMetadata saved to: {metadata_path}")
print(f"Error report saved to: {errors_path}")
print("=================================================")