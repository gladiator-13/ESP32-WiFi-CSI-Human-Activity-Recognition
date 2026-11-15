
from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_DIR = REPORTS_DIR / "recording_level"

MODELS = ["random_forest", "svm", "cnn"]

MODEL_COLUMNS = {
    "within_location": {
        "random_forest": {
            "actual": "actual_activity",
            "predicted": "predicted_activity",
        },
        "svm": {
            "actual": "true_label",
            "predicted": "predicted_label",
        },
        "cnn": {
            "actual": "actual_label",
            "predicted": "predicted_label",
        },
    },
    "lolo": {
        "random_forest": {
            "actual": "actual_label",
            "predicted": "predicted_label",
        },
        "svm": {
            "actual": "actual_label",
            "predicted": "predicted_label",
        },
        "cnn": {
            "actual": "actual_label",
            "predicted": "predicted_label",
        },
    },
}

LABELS = list(range(1, 21))

CLASS_NAMES = [
    f"Activity_{i:02d}"
    for i in LABELS
]


def get_prediction_path(model, evaluation, fold=None):

    if evaluation == "within_location":

        filename = "test_predictions.csv"

        if model == "cnn":
            filename = "test_predictions_with_metadata.csv"

        return (
            REPORTS_DIR
            / model
            / "within_location"
            / filename
        )

    return (
        REPORTS_DIR
        / model
        / "lolo"
        / f"fold_{fold}"
        / "test_predictions.csv"
    )


def aggregate_recordings(df, model):

    evaluation = "within_location" if "actual_activity" in df.columns or "true_label" in df.columns else "lolo"

    actual_col = MODEL_COLUMNS[evaluation][model]["actual"]
    predicted_col = MODEL_COLUMNS[evaluation][model]["predicted"]

    # Each recording must have exactly one true activity.
    true_counts = (
        df.groupby("recording_id")[actual_col]
        .nunique()
    )

    assert (true_counts == 1).all()

    results = []
    total_ties = 0

    for recording_id, group in df.groupby(
        "recording_id",
        sort=True,
    ):

        assert len(group) == 19

        actual_label = int(group[actual_col].iloc[0])

        votes = (
            group[predicted_col]
            .astype(int)
            .value_counts()
        )

        max_votes = votes.max()

        winners = sorted(
            votes[votes == max_votes].index.tolist()
        )

        tie = len(winners) > 1

        if tie:
            total_ties += 1

        # Deterministic tie-break:
        # choose the smallest numeric activity label.
        predicted_label = int(winners[0])

        results.append({
            "recording_id": recording_id,
            "participant": group["participant"].iloc[0],
            "location": group["location"].iloc[0],
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "winning_votes": int(max_votes),
            "total_windows": len(group),
            "tie": tie,
        })

    return pd.DataFrame(results), total_ties


def calculate_metrics(recordings):

    y_true = recordings["actual_label"]
    y_pred = recordings["predicted_label"]

    metrics = {
        "recording_count": int(len(recordings)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def save_evaluation(recordings, model, evaluation, fold=None):

    if evaluation == "within_location":
        output_path = (
            OUTPUT_DIR
            / "within_location"
            / model
        )
    else:
        output_path = (
            OUTPUT_DIR
            / "lolo"
            / f"fold_{fold}"
            / model
        )

    output_path.mkdir(parents=True, exist_ok=True)

    # Recording-level predictions
    recordings.to_csv(
        output_path / "recording_predictions.csv",
        index=False,
    )

    # Metrics
    metrics = calculate_metrics(recordings)

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Per-class metrics
    report = classification_report(
        recordings["actual_label"],
        recordings["predicted_label"],
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_path / "classification_report.csv"
    )

    # Confusion matrix
    matrix = confusion_matrix(
        recordings["actual_label"],
        recordings["predicted_label"],
        labels=LABELS,
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(output_path / "confusion_matrix.csv")

    return metrics


def evaluate_one(model, evaluation, fold=None):

    path = get_prediction_path(model, evaluation, fold)

    df = pd.read_csv(path)

    recordings, tie_count = aggregate_recordings(df, model)

    metrics = save_evaluation(
        recordings,
        model,
        evaluation,
        fold,
    )

    metrics["model"] = model
    metrics["evaluation"] = evaluation
    metrics["fold"] = fold
    metrics["tie_count"] = tie_count

    print(
        f"PASS | {model:15s} | "
        f"Recordings: {len(recordings):3d} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"Ties: {tie_count}"
    )

    return metrics


def main():

    print("=" * 75)
    print("RECORDING-LEVEL EVALUATION")
    print("=" * 75)

    all_results = []

    # Within-location evaluation
    print("\nWITHIN-LOCATION")

    for model in MODELS:

        result = evaluate_one(
            model,
            "within_location",
        )

        all_results.append(result)

    # LOLO evaluation
    for fold in range(1, 4):

        print(f"\nLOLO FOLD {fold}")

        for model in MODELS:

            result = evaluate_one(
                model,
                "lolo",
                fold,
            )

            all_results.append(result)

    summary = pd.DataFrame(all_results)

    summary_dir = OUTPUT_DIR / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        summary_dir / "recording_level_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("RECORDING-LEVEL SUMMARY")
    print("=" * 75)

    print(
        summary[
            [
                "model",
                "evaluation",
                "fold",
                "recording_count",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "tie_count",
            ]
        ].to_string(index=False)
    )

    print("\nAll recording-level evaluations completed successfully.")


if __name__ == "__main__":
    main()
