
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"

MODELS = ["random_forest", "svm", "cnn"]


def get_prediction_path(model, evaluation, fold=None):

    if evaluation == "within_location":
        filename = "test_predictions.csv"

        if model == "cnn":
            filename = "test_predictions_with_metadata.csv"

        return (
            REPORTS_DIR / model / "within_location" / filename
        )

    return (
        REPORTS_DIR
        / model
        / "lolo"
        / f"fold_{fold}"
        / "test_predictions.csv"
    )



def validate_file(path, model):

    df = pd.read_csv(path)

    # Detect model-specific label columns
    actual_column = next(
        (
            column
            for column in ["actual_label", "actual_activity", "true_label"]
            if column in df.columns
        ),
        None,
    )

    predicted_column = next(
        (
            column
            for column in ["predicted_label", "predicted_activity"]
            if column in df.columns
        ),
        None,
    )

    required_columns = [
        "recording_id",
        "window_number",
    ]

    assert actual_column is not None, (
        f"Actual label column missing in {path}"
    )

    assert predicted_column is not None, (
        f"Predicted label column missing in {path}"
    )

    required_columns += [actual_column, predicted_column]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    assert not missing, f"Missing columns: {missing}"

    assert not df[required_columns].isnull().any().any(), (
        "Missing values found in required columns."
    )

    assert not df.duplicated(
        ["recording_id", "window_number"]
    ).any(), "Duplicate recording-window pairs found."

    counts = df.groupby("recording_id").size()

    assert (counts == 19).all(), (
        "Some recordings do not have exactly 19 windows."
    )

    inconsistent = (
        df.groupby("recording_id")[actual_column].nunique()
    )

    assert (inconsistent == 1).all(), (
        "Inconsistent actual labels within a recording."
    )

    assert df["window_number"].between(1, 19).all(), (
        "Invalid window numbers."
    )

    print(
        f"PASS | {model:15s} | "
        f"Windows: {len(df):4d} | "
        f"Recordings: {len(counts):3d}"
    )



def main():

    print("=" * 65)
    print("RECORDING-LEVEL INPUT VALIDATION")
    print("=" * 65)

    total = 0

    for model in MODELS:

        path = get_prediction_path(model, "within_location")
        validate_file(path, model + " within")
        total += 1

    for fold in range(1, 4):

        print(f"\nLOLO FOLD {fold}")

        for model in MODELS:

            path = get_prediction_path(
                model,
                "lolo",
                fold,
            )

            validate_file(path, model)
            total += 1

    print("\n" + "=" * 65)
    print(f"ALL {total} DATASETS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
