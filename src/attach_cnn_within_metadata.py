
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    BASE_DIR
    / "reports/cnn/within_location/test_predictions.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "data/processed/normalized/within_location/test/metadata.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "reports/cnn/within_location/test_predictions_with_metadata.csv"
)


def main():

    print("Loading CNN predictions and metadata...")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # Verify row counts
    assert len(predictions) == len(metadata), (
        "Prediction and metadata row counts do not match."
    )

    # Verify actual activity labels
    assert (
        predictions["actual_label"].to_numpy()
        == metadata["activity"].to_numpy()
    ).all(), "Actual activity labels do not match."

    # Verify activity names
    expected_names = [
        f"Activity_{activity:02d}"
        for activity in metadata["activity"]
    ]

    assert (
        predictions["actual_activity"].to_numpy()
        == expected_names
    ).all(), "Activity names do not match."

    # Attach metadata without duplicating existing columns
    metadata_columns = [
        column
        for column in metadata.columns
        if column not in predictions.columns
    ]

    augmented = pd.concat(
        [
            predictions.reset_index(drop=True),
            metadata[metadata_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    # Save augmented predictions
    augmented.to_csv(OUTPUT_PATH, index=False)

    print("\nMETADATA ATTACHMENT SUCCESSFUL")
    print("--------------------------------")
    print("Prediction rows:", len(augmented))
    print("Attached columns:", metadata_columns)
    print("Output:", OUTPUT_PATH)
    print("Final columns:", augmented.columns.tolist())


if __name__ == "__main__":
    main()
