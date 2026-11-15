
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SIGNALS_DIR = PROJECT_ROOT / "data" / "processed" / "signals"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "windows"

WINDOW_SIZE = 200       # 2 seconds at 100 Hz
STRIDE = 100            # 50% overlap
EXPECTED_SHAPE = (2000, 52)

ACTIVITY_CLASSES = list(range(1, 21))


# ==================================================
# 2. LOAD AND VALIDATE RECORDING SPLITS
# ==================================================

def load_split_file(path, is_lolo=False):

    print(f"\nLoading: {path.name}")

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "recording_id",
        "participant",
        "activity",
        "recording_type",
        "location",
        "split",
    }

    if is_lolo:
        required.add("fold")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Primary experiment: only activities 1-20
    df = df[df["activity"].isin(ACTIVITY_CLASSES)].copy()

    if df.empty:
        raise ValueError("No activity recordings found.")

    return df


# ==================================================
# 3. LOAD ONE PREPROCESSED RECORDING
# ==================================================

def load_signal(recording_id):

    recording_id = str(recording_id)

    # Metadata normally stores the recording ID without extension
    signal_path = SIGNALS_DIR / f"{recording_id}.npy"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {signal_path}"
        )

    signal = np.load(signal_path, mmap_mode="r")

    if signal.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{recording_id}: expected {EXPECTED_SHAPE}, "
            f"got {signal.shape}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            f"{recording_id}: signal contains NaN or infinity."
        )

    return signal


# ==================================================
# 4. CALCULATE NUMBER OF WINDOWS
# ==================================================

def calculate_num_windows(signal_length):

    if signal_length < WINDOW_SIZE:
        return 0

    return (
        (signal_length - WINDOW_SIZE) // STRIDE
    ) + 1


# ==================================================
# 5. GENERATE WINDOWS FOR ONE SPLIT
# ==================================================

def generate_split_windows(df, output_path, split_name):

    output_path.mkdir(parents=True, exist_ok=True)

    df = df.reset_index(drop=True)

    # Determine expected number of windows
    total_windows = 0

    for _, row in df.iterrows():

        signal = load_signal(row["recording_id"])

        total_windows += calculate_num_windows(
            signal.shape[0]
        )

    if total_windows == 0:
        raise ValueError(
            f"No windows generated for {split_name}."
        )

    # Create memory-mapped arrays to avoid loading
    # the complete dataset into RAM.
    X = np.lib.format.open_memmap(
        output_path / "X.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, WINDOW_SIZE, EXPECTED_SHAPE[1]),
    )

    y = np.lib.format.open_memmap(
        output_path / "y.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_windows,),
    )

    window_metadata = []
    window_index = 0

    for _, row in df.iterrows():

        recording_id = str(row["recording_id"])
        signal = load_signal(recording_id)

        activity = int(row["activity"])

        num_windows = calculate_num_windows(
            signal.shape[0]
        )

        for window_number in range(num_windows):

            start = window_number * STRIDE
            end = start + WINDOW_SIZE

            window = signal[start:end]

            if window.shape != (
                WINDOW_SIZE,
                EXPECTED_SHAPE[1],
            ):
                raise ValueError(
                    f"Invalid window shape in {recording_id}"
                )

            X[window_index] = window
            y[window_index] = activity

            metadata_row = {
                "window_id": f"{recording_id}_W{window_number + 1:02d}",
                "recording_id": recording_id,
                "participant": row["participant"],
                "location": row["location"],
                "activity": activity,
                "split": split_name,
                "window_number": window_number + 1,
                "start_sample": start,
                "end_sample": end,
            }

            if "fold" in row.index:
                metadata_row["fold"] = int(row["fold"])

            window_metadata.append(metadata_row)

            window_index += 1

    X.flush()
    y.flush()

    del X
    del y

    metadata_df = pd.DataFrame(window_metadata)

    metadata_df.to_csv(
        output_path / "metadata.csv",
        index=False,
    )

    if window_index != total_windows:
        raise ValueError(
            f"Expected {total_windows} windows, "
            f"generated {window_index}."
        )

    print(
        f"{split_name}: {window_index} windows generated "
        f"| X shape: ({window_index}, {WINDOW_SIZE}, 52)"
    )

    return window_index


# ==================================================
# 6. GENERATE ONE COMPLETE EVALUATION DATASET
# ==================================================

def generate_dataset(df, output_path, dataset_name):

    print("\n" + "=" * 55)
    print(f"GENERATING: {dataset_name}")
    print("=" * 55)

    total = 0

    for split_name in ["train", "validation", "test"]:

        split_df = df[df["split"] == split_name]

        if split_df.empty:
            raise ValueError(
                f"{dataset_name}: {split_name} is empty."
            )

        count = generate_split_windows(
            split_df,
            output_path / split_name,
            split_name,
        )

        total += count

    print(f"\nTotal windows for {dataset_name}: {total}")

    return total


# ==================================================
# 7. WITHIN-LOCATION WINDOW GENERATION
# ==================================================

def generate_within_location():

    print("\n\nWITHIN-LOCATION DATASET")

    split_file = SPLITS_DIR / "recording_splits.csv"

    df = load_split_file(split_file)

    return generate_dataset(
        df,
        OUTPUT_DIR / "within_location",
        "Within-location",
    )


# ==================================================
# 8. LOLO WINDOW GENERATION
# ==================================================

def generate_lolo():

    print("\n\nLEAVE-ONE-LOCATION-OUT DATASETS")

    split_file = SPLITS_DIR / "lolo_recording_splits.csv"

    df = load_split_file(split_file, is_lolo=True)

    total = 0

    for fold_number in sorted(df["fold"].unique()):

        fold_df = df[
            df["fold"] == fold_number
        ].copy()

        test_location = fold_df["test_location"].iloc[0]

        print(
            f"\nFold {fold_number} | "
            f"Held-out location: {test_location}"
        )

        fold_output = (
            OUTPUT_DIR / "lolo" / f"fold_{fold_number}"
        )

        count = generate_dataset(
            fold_df,
            fold_output,
            f"LOLO Fold {fold_number}",
        )

        total += count

    print(f"\nTotal windows across all LOLO folds: {total}")


# ==================================================
# 9. MAIN
# ==================================================

def main():

    print("=" * 55)
    print("CSI WINDOW GENERATION")
    print("=" * 55)

    print(f"Window size: {WINDOW_SIZE} samples")
    print(f"Stride: {STRIDE} samples")
    print(f"Overlap: 50%")
    print(f"Expected recording shape: {EXPECTED_SHAPE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate both evaluation strategies
    generate_within_location()
    generate_lolo()

    print("\n" + "=" * 55)
    print("WINDOW GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 55)


if __name__ == "__main__":
    main()
